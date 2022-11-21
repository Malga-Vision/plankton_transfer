from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from termcolor import colored
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from imagedataset import AdvancedImageFolder, PartitionDistributedSampler

from .composed import ComposedModel


from utils import load_config, optimizer_by_name, scheduler_by_name


class BasicFineTuner:


    def __init__(self,
                 model: ComposedModel,
                 ckpt_name: str,
                 config_dict: Optional[dict] = None,
                 batch_size: Optional[int] = None,
                 n_accumulation: Optional[int] = None,
                 verbose: Optional[bool] = True):

        """
            Initialize a FineTuner.

            Args:
                model (ComposedModel): the ComposedModel to finetune.
                ckpt_name (str): the name of the ckpt file.
                classifier).
                config_dict (str, optional): all hparams in a dict.

            Note:
                One and just one betweene config_path and config_dict should be passed.

            Raises:
                ValueError if zero or two between config_path and config_dict are passed.

"""

        
        self.ckpt_name = ckpt_name
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        
        # only rank 0 can have verbose=True
        verbose = False if self.rank != 0 else verbose

        # set hparams as attributes
        for k in config_dict:
            setattr(self, k, config_dict[k])

        # get batch size
        if batch_size is not None and n_accumulation is not None:
            self.batch_size = batch_size
            self.n_accumulation = n_accumulation

            if verbose:
                print(colored("-" * 30, "red"))
                print(colored(f"Batch size from args", "red"))
                print(f"GPU_BS:       {self.batch_size} (x {self.world_size} GPUs)")
                print(f"ACCUMULATION: {self.n_accumulation}")
                print(colored("-" * 30, "red"))

        else:
            if verbose:
                print(colored("-" * 30, "red"))
                print(colored(f"Batch size from config", "red"))
                print(f"GPU_BS:       {self.batch_size} (x {self.world_size} GPUs)")
                print(f"ACCUMULATION: {self.n_accumulation}")
                print(colored("-" * 30, "red"))


        self._model = model.to(self.local_rank)

        if self.sync_bn:
            self._model = nn.SyncBatchNorm.convert_sync_batchnorm(self._model)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)


    def warmup(self,
               dataset_train: AdvancedImageFolder,
               dataset_eval: Optional[AdvancedImageFolder] = None,
               verbose: Optional[bool] = False) -> Union[float, None]:

        """
            Warms up scheduled into two steps.
            In the first step the classifier and bottleneck are trained, while the 
            backbone is freezed.
            In the second step the backbone is tuned with a lr 10 times smaller than 
            the one that will be used during training.

            Args:
                dataset_train (AdvanceImageFolder): the dataset for the training.
                verbose (bool, optional): True to print debug logs on standard output.

        """
        # only rank 0 can have verbose=True
        verbose = False if self.rank != 0 else verbose

        sampler = DistributedSampler(dataset_train, drop_last=True)

        loader = dataset_train.cudaloader(batch_size=self.batch_size,
                                          drop_last=True,
                                          num_workers=self.num_workers,
                                          pin_memory=self.pin_memory,
                                          sampler=sampler)


        warmup_steps = [("STEP 1", 0., self.wu_epochs_step1), 
                        ("STEP 2", self.lr_backbone_wu, self.wu_epochs_step2)]

        for step_name, lr_backbone, epochs in warmup_steps:

            ddp_model = None

            # NOTE: DDP set requires_grad to false when a paramter have 0 lr, so we 
            # reset back to true if in previous step that was freezed.

            for p in self._model.parameters(): p.requires_grad = True

            scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

            # OPTIMIZER
            param_groups = self._model.get_param_groups(lr_backbone=lr_backbone, 
                                                    lr_classifier=self.lr_classifier_wu, 
                                                    lr_bottleneck=self.lr_bottleneck_wu)

            optimizer = optimizer_by_name(self.optimizer,
                                          param_groups=param_groups,
                                          hparams_dict=self.optimizer_params)

            ddp_model = DDP(self._model,
                            device_ids=[self.local_rank],
                            output_device=self.local_rank,
                            find_unused_parameters=False)

            # steps for one epoch
            update_steps_one_epoch = len(loader)//self.n_accumulation

            # initialize crossentropyloss (from logits).
            criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)\
                        .to(self.local_rank)

            # if verbose:
            #     for p in self._model.backbone.parameters():
            #         print(p.requires_grad)

            for epoch in range(epochs):
            
                number_updates = 0
                sampler.set_epoch(epoch)
                ddp_model.train()

                if verbose:
                    start_epoch_time = time.time()
                    print("FINETUNER -", end=" ")
                    print(colored(f"WARM UP {step_name} EPOCH {epoch+1}/{epochs}", "cyan"), end=": ")

                # loop for current epoch
                for step, batch in enumerate(loader):
                
                    # reached maximum number of updates
                    if number_updates >= update_steps_one_epoch:
                        optimizer.zero_grad(set_to_none=True)
                        break

                    images, labels = batch.image, batch.label

                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        _, _, out = ddp_model(images)
                        loss = criterion(out, labels)/self.n_accumulation

                    scaler.scale(loss).backward()

                    if (step + 1) % self.n_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        number_updates += 1

    
                # current epoch finished
                if verbose:
                    end_epoch_time = time.time()
                    total_time = end_epoch_time - start_epoch_time
                    print(f"({total_time:.2f}s)", end=" ")

                # Validation
                if dataset_eval:
                    # validation
                    accuracy = self._eval(ddp_model=ddp_model, 
                                          dataset_eval=dataset_eval, 
                                          max_samples=self.max_eval_samples, 
                                          epoch=epoch)
                    
                    
                    # print info if needed
                    if verbose:
                        end_val_time = time.time()
                        total_time = end_val_time - end_epoch_time
                        print(colored(f"VACC = {accuracy:.3f}", "blue"), end=" ")
                        print(f"({total_time:.2f}s)", end=" ")

                # new line after ending epoch
                if verbose:
                    print()


    def fit(self,
            dataset_train: AdvancedImageFolder,
            dataset_eval: Optional[AdvancedImageFolder] = None,
            verbose: Optional[bool] = False) -> Union[float, None]:

        """
            Fits the model to the dataset_train. Additionally a validation dataset can
            used to select the right stopping time.

            Args:
                dataset_train (AdvanceImageFolder): the dataset for the training.
                dataset_eval (AdvanceImageFolde, optioanl): the dataset for the
                validation.
                verbose (bool, optional): True to print debug logs on standard output.

            Returns:
                the validation accuracy of best epoch (of saved model) if eval dataset
                is not None, None otherwise.

            Note:
                it saves on disk the best performing checkpoint across epochs.
        """
        # only rank 0 can have verbose=True
        verbose = False if self.rank != 0 else verbose

        sampler = DistributedSampler(dataset_train, drop_last=True)

        loader = dataset_train.cudaloader(batch_size=self.batch_size,
                                          drop_last=True,
                                          num_workers=self.num_workers,
                                          pin_memory=self.pin_memory,
                                          sampler=sampler)

        ddp_model = None
        # OPTIMIZER
        param_groups = self._model.get_param_groups(lr_backbone=self.lr_backbone, 
                                                    lr_classifier=self.lr_classifier, 
                                                    lr_bottleneck=self.lr_bottleneck)

        
        optimizer = optimizer_by_name(self.optimizer,
                                      param_groups=param_groups,
                                      hparams_dict=self.optimizer_params)


        ddp_model = DDP(self._model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False)


        # steps for one epoch
        update_steps_one_epoch = len(loader)//self.n_accumulation

        if self.max_update_steps_one_epoch > 0:
            update_steps_one_epoch = min([self.max_update_steps_one_epoch, 
                                          update_steps_one_epoch])

        # SCHEDULER
        total_update_steps = self.max_epochs * update_steps_one_epoch

        scheduler = scheduler_by_name(self.scheduler,
                                      optimizer=optimizer,
                                      hparams_dict=self.scheduler_params,
                                      num_steps=total_update_steps)

        # initialize crossentropyloss (from logits).
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)\
                    .to(self.local_rank)

        # Training
        best_accuracy = 0.0 if dataset_eval is not None else None
        not_improving_eval = 0
        
        for epoch in range(self.max_epochs):
            
            number_updates = 0
            sampler.set_epoch(epoch)
            ddp_model.train()

            if verbose:
                start_epoch_time = time.time()

                print("FINETUNER -", end=" ")
                print(colored(f"EPOCH {epoch+1}/{self.max_epochs}", "red"), end=": ")

            # loop for current epoch
            for step, batch in enumerate(loader):
                
                # reached maximum number of updates
                if number_updates >= update_steps_one_epoch:
                    optimizer.zero_grad(set_to_none=True)
                    break

                images, labels = batch.image, batch.label

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _, _, out = ddp_model(images)
                    loss = criterion(out, labels)/self.n_accumulation

                self.scaler.scale(loss).backward()

                if (step + 1) % self.n_accumulation == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    number_updates += 1

                    # use a step scheduler
                    if self.step_scheduler:
                        scheduler.step()

            # use a epoch scheduler
            if not self.step_scheduler:
                scheduler.step()

            # current epoch finished
            if verbose:
                end_epoch_time = time.time()
                total_time = end_epoch_time - start_epoch_time
                print(f"({total_time:.2f}s)", end=" ")

            # Validation
            if dataset_eval and self.eval_period > 0:

                # if we need to evaluate or if it is last epoch
                if ((epoch + 1) == self.max_epochs) or (
                    (epoch + 1) % self.eval_period == 0
                ):

                    # validation
                    accuracy = self._eval(ddp_model=ddp_model, 
                                         dataset_eval=dataset_eval, 
                                         max_samples=self.max_eval_samples, 
                                         epoch=epoch)
                    
                    
                    # print info if needed
                    if verbose:
                        end_val_time = time.time()
                        total_time = end_val_time - end_epoch_time
                        print(colored(f"VACC = {accuracy:.3f}", "blue"), end=" ")
                        print(f"({total_time:.2f}s)", end=" ")

                    # check if we improved
                    if accuracy > best_accuracy:
                        if verbose:
                            print(
                            colored(f"SAVED (OLD_BEST = {best_accuracy:.3f})", "green"),
                            end=" ")

                        not_improving_eval = 0
                        best_accuracy = accuracy
                        self._save()

                    elif epoch >= self.min_epochs:
                        not_improving_eval += 1


                    # a lot of epochs without improvement, end training.
                    if not_improving_eval >= self.stop_after_not_improving_eval:
                        if verbose:
                            print()
                            print(colored(f"## STOPPING: {not_improving_eval} not " +
                                           "improving validations! ##", "green"))
                            print()
                            
                        return best_accuracy
            else:
                # we not have validation set, save model at every iteration!
                self._save()

            # new line after ending epoch
            if verbose:
                print()

        return best_accuracy


    def load(self, ckpt_path, load_classifier=False) -> BasicFineTuner:
        """ Loads ckpt from path. """

        map_location = {"cuda:%d" % 0: "cuda:%d" % self.local_rank}
        state_dict = torch.load(ckpt_path, map_location=map_location)

        if not load_classifier:
            for k in list(state_dict.keys()):
                if "classifier" in k:
                    del state_dict[k]

        self._model.load_state_dict(state_dict, strict=False)

        return self


    def _save(self) -> None:
        """ Saves the current model ckpt. """

        dist.barrier()
        if self.rank == 0:
            path = self.ckpt_name
            torch.save(self._model.state_dict(), path)
        dist.barrier()


    def eval(self, dataset_eval: AdvancedImageFolder):
        ddp_model = DDP(self._model,
                        device_ids=[self.local_rank],
                        output_device=self.local_rank,
                        find_unused_parameters=False).to(self.local_rank)
        return self._eval(ddp_model, dataset_eval)


    def _eval(self,
             ddp_model,
             dataset_eval: AdvancedImageFolder, 
             max_samples: Optional[int] = -1,
             epoch: Optional[int] = 0) -> float:

        ddp_model.eval()

        sampler = PartitionDistributedSampler(dataset_eval, shuffle=True)
        sampler.set_epoch(epoch)
        
        # DATALOADER
        loader = dataset_eval.cudaloader(sampler=sampler,
                                         num_workers=self.num_workers,
                                         drop_last=False, 
                                         batch_size=self.batch_size)


        if max_samples > 0 and max_samples < len(dataset_eval):
            n_batches = max_samples // (self.batch_size * dist.get_world_size())
        else:
            n_batches = len(loader)

        # counter for all correct predictions.
        corrects = torch.zeros(1, device=self.local_rank)
        samples  = torch.zeros(1, device=self.local_rank)
        
        for n, batch in enumerate(loader):

            if n >= n_batches:
                break

            images = batch.image
            labels = batch.label

            with torch.no_grad():
                _, _, out = ddp_model(images)
                _, argmax = torch.max(out, dim=1)
                corrects += torch.sum(argmax == labels)
                samples += len(images)

        dist.all_reduce(corrects)
        dist.all_reduce(samples)

        accuracy = float(corrects.cpu()/samples.cpu())

        ddp_model.train()

        return accuracy
