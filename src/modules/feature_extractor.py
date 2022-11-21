from typing import Dict, Optional, Tuple, Union

import time
import os
from termcolor import colored

import torch.distributed as dist
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from imagedataset import AdvancedImageFolder, MultiDataset, PartitionDistributedSampler
from modules.composed import ComposedModel

from .backbone import Backbone
from .utility_functions import get_output_dim


def extract_features(model: Union[Backbone, ComposedModel],
                     datasets: Union[AdvancedImageFolder, MultiDataset],
                     batch_size: int,
                     features_dim: Optional[int] = None,
                     num_workers: Optional[int] = 4,
                     return_logits: Optional[bool] = False,
                     pin_memory: Optional[bool] = True,
                     verbose: Optional[bool] = False)\
                     -> Union[Tuple[torch.Tensor, torch.Tensor],
                              Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                              Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                              Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
        Extract the features using a model.

        Note 1:
            The output has 'same format' as input.
            - AdvancedImageFolder -> Single Tuple
            - MultiDataset        -> Dict of Tuples

        Note 2:
            Tuples are (features tensor, labels tensor) if return_logits == False
            or (features tensor, labels tensor, logits tensor) if return_logits == True

        Note 3:
            If the output of a model is a Tuple, than we assume logits are the last dim,
            while fetaures are the last but one dimension.

        Note 4:
            the dataset will NOT be shuffled, so features/labels are in the original
            dataset order.

        Args:
            model (torch.module): the feature extractor.
            datasets (dict, list or AdvanceImageFolder): the datasets to process.
            batch_size (int): the batch size for inference.
            features_dim (int, optional): the features dimension. If None it is assumed
            the model is a timm feature extractor and the dimension will be taken
            automatically.
            parallel (bool, optional): True to use data parallel.
            num_workers (int, optional): the number of workers for the dataloader.
            return_logits (bool, optional): True to return also logits.
            pin_memory (bool, optional): True to pin memory of dataloader.
            verbose (bool, optional): True to print debug info, False otherwise.
            enable_tqdm (bool, optional): True to enable tqdm on the loop on dataloader.

        Returns:
            a dictionary, a list or a single Tuple of tensors (depending on input)


        Raises:
            WrongOutputError if the model output is of a wrong type.
            ValueError if return logits is True but model returns just features.
    """

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    if verbose:
        print(f"Rank {rank+1}/{world_size} started.")

    if not isinstance(model, DDP):
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    model.eval()

    outputs = {}

    if isinstance(datasets, AdvancedImageFolder):
        datasets_dict = {datasets.dataset_name: datasets}
    elif isinstance(datasets, MultiDataset):
        datasets_dict = datasets
    else:
        raise ValueError("'datasets' should be an AdvanceImageFolder or a Multidataset")

    for name, dataset in datasets_dict.items():

        sampler = PartitionDistributedSampler(dataset)
        partitions = sampler.get_partitions()
        partition_length = len(partitions[0])

        if verbose:

            for i in range(world_size):
                dist.barrier()
                if i == rank:
                    print(
                        f"Rank {rank}/{world_size}: indices from "
                        + f"{min(partitions[rank])} to {max(partitions[rank])}."
                    )

        # info of current dataset
        n_images = partition_length
        n_classes = dataset.n_classes()

        if features_dim is None and isinstance(model, ComposedModel):
            features_dim = model.get
        elif features_dim is None:
            # This can raise WrongOutputError
            features_dim = get_output_dim(model)

        # tensors to store features and labels.
        features = torch.empty([n_images, features_dim], device=local_rank).float()
        labels = torch.zeros(n_images).to(local_rank).long()

        # tensor to collect features and labels from all ranks.
        features_global = [
            torch.zeros([partition_length, features_dim], device=local_rank)
            for _ in range(world_size)
        ]
        labels_global = [
            torch.zeros(partition_length).to(local_rank).long()
            for _ in range(world_size)
        ]

        # tensors for optional logits.
        if return_logits:
            logits = torch.zeros([n_images, n_classes], device=local_rank).float()
            logits_global = [
                torch.zeros([partition_length, n_classes], device=local_rank).float()
                for _ in range(world_size)
            ]

        loader = dataset.cudaloader(batch_size=batch_size,
                                    sampler=sampler,
                                    num_workers=num_workers,
                                    drop_last=False,
                                    pin_memory=pin_memory, 
                                    rank=local_rank)

        start = time.time()

        for i, batch in enumerate(loader):

            batch_images, batch_labels = batch.image, batch.label
            batch_images = batch_images

            with torch.no_grad():
                batch_out = model(batch_images)

                if isinstance(batch_out, tuple): # case ComposedModel
                    batch_feats = batch_out[-2]
                    batch_logits = batch_out[-1]
                else: # case backbone
                    batch_feats = batch_out

                    if return_logits:
                        raise ValueError(
                            "Asked to return logits but model has just one output..."
                        )

            # saving features and labels...
            starting_index = i * batch_size
            ending_index = starting_index + batch_feats.shape[0]

            features[starting_index:ending_index, :] = batch_feats
            labels[starting_index:ending_index] = batch_labels

            if return_logits:
                logits[starting_index:ending_index] = batch_logits

        end = time.time()

        # collect all features/labels from ranks
        dist.all_gather(features_global, features)
        dist.all_gather(labels_global, labels)

        # concatenate global features/labels
        features = torch.cat(features_global)[: len(dataset)]
        labels = torch.cat(labels_global)[: len(dataset)]

        # get logits from all ranks and concatenate them
        if return_logits:
            dist.all_gather(logits_global, logits)
            logits = torch.cat(logits_global)[: len(dataset)]

        outputs[name] = (
            (features, labels, logits) if return_logits else (features, labels)
        )

        if verbose and local_rank == 0:
            print(colored(f"Time to extract features: {end-start:6.4}s", "green"))
            print()

    # set outputs to correct format
    if isinstance(datasets, AdvancedImageFolder):
        outputs = outputs[datasets.dataset_name]

    return outputs
