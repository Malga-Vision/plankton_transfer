import argparse
from utils import load_config
from termcolor import colored
import pandas as pd

import os
from os.path import join
from pathlib import Path

from modules import BasicFineTuner, get_composed_model, get_dataset_for_model
import torch.distributed as dist
dist.init_process_group("nccl")

import torch
torch.backends.cudnn.benchmark = False

CKPT_FOLDER = "outputs/ckpt/"
OUTPUT_DEFAULT = "outputs/"
OUTPUT_NAME = "finetuning_output.csv"
CONFIG_ARCHITECTURE = "config/architecture.yaml"
CONFIG_TRAINING = "config/training.yaml"


parser = argparse.ArgumentParser(description='Finetuning Parser')
parser.add_argument('--dataset_train',    type=str)
parser.add_argument('--dataset_test',     type=str)
parser.add_argument('--ckpt_load',        type=str)
parser.add_argument('--ckpt_save',        type=str)
parser.add_argument('--model_name',       type=str)
parser.add_argument('--batch_size',       type=int)
parser.add_argument('--accumulation',     type=int)



def save_results(log_path, log_name, results_df):
    """
        Save the results in a csv file. If the file already exists the results are 
        appended to the file (as new rows).

        Args:
            log_path (string): the path to the log directory.
            log_name(string): the name of the log file.
            results_dict (pd.DataFrame): the results in a pd.DataFrame
    """

    # create directory if it does not exist.
    Path(log_path).mkdir(parents=True, exist_ok=True)

    if not log_name.endswith(".csv"):
        log_name += ".csv"

    log_full_path = os.path.join(log_path, log_name)

    # try append row
    try:
        old = pd.read_csv(log_full_path)
        new = pd.concat([old, results_df])
    except:
        new = results_df
    
    # save new dataframe
    new.to_csv(log_full_path, encoding='utf-8', index=False)


def main(model_name, dataset_train, dataset_test, ckpt_load, batch_size, accumulation, ckpt_save):


    Path(CKPT_FOLDER).mkdir(exist_ok=True)

    ckpt_save = join(CKPT_FOLDER, ckpt_save)
    if ckpt_load:
        ckpt_load = join(CKPT_FOLDER, ckpt_load)

    # only rank 0 is verbose (to avoid multiple print)
    verbose = dist.get_rank() == 0

    # get hparams
    hparams_architecture = load_config(CONFIG_ARCHITECTURE)
    hparams_training     = load_config(CONFIG_TRAINING)

    # extract hparams for architecture
    color_jitter    = hparams_architecture["color_jitter"]
    random_hflip    = hparams_architecture["random_hflip"]
    random_vflip    = hparams_architecture["random_vflip"]
    drop_rate       = hparams_architecture["drop_rate"]
    drop_path_rate  = hparams_architecture["drop_path_rate"]

    bottleneck_dim  = hparams_architecture["bottleneck_dim"]
    classifier_bias = hparams_architecture["classifier_bias"]
    classifier_wn   = hparams_architecture["classifier_wn"]
    
    if verbose:
        print()
        print(colored("================= ARCHITECTURE INFO ==================", "green"))
        print(f"Backbone: {model_name}")
        print(f"Bottleneck dimension: {bottleneck_dim}")
        print(f"Use classifier bias: {classifier_bias}")
        print(f"Use classifier weight norm: {classifier_wn}" )
        print(colored("================= DATASET/CKPT INFO ==================", "green"))
        print(f"Train dataset: {dataset_train}")
        print(f"Test dataset: {dataset_test if dataset_test is not None else dataset_train}")
        print(f"Load ckpt from: {ckpt_load}")
        print(f"Save ckpt to: {ckpt_save}")
        print(colored("======================================================", "green"))
        print()

    # get dataset 
    train_dataset = get_dataset_for_model(model_name, dataset_train)

    if dataset_test is None:
        if verbose:
            print(f"Test dataset not selected!")

        test_dataset = None 

    elif dataset_test == dataset_train:
        if verbose:
            print(f"Splitting {dataset_train} into train - test: 80% - 20%")

        train_dataset, test_dataset = train_dataset.random_split([0.8, 0.2], 
                                                                 ["train", "test"])
    else:
        test_dataset =  get_dataset_for_model(model_name, dataset_test)

    train_dataset, eval_dataset = train_dataset.random_split([0.85, 0.15], ["train", "eval"])
    # eval_dataset = test_dataset

    n_classes = train_dataset.n_classes()

    # get model  
    model = get_composed_model(backbone_name=model_name, 
                               pretrained=True, 
                               color_jitter_train=color_jitter, 
                               random_hflip=random_hflip,
                               random_vflip=random_vflip,
                               n_classes=n_classes, 
                               bottleneck_dim=bottleneck_dim, 
                               classifier_bias=classifier_bias, 
                               classifier_wn=classifier_wn,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate)


    finetuner = BasicFineTuner(model=model,
                               ckpt_name=ckpt_save,
                               config_dict=hparams_training,
                               batch_size=batch_size,
                               n_accumulation=accumulation)


    if ckpt_load is not None:
        if verbose:
            print(f"Loading ckpt from: {ckpt_load}.")

        finetuner.load(ckpt_load, load_classifier=False)

    finetuner.warmup(dataset_train=train_dataset, 
                     dataset_eval=eval_dataset, 
                     verbose=True)

    finetuner.fit(dataset_train=train_dataset, 
                  dataset_eval=eval_dataset, 
                  verbose=True)

    accuracy = finetuner.eval(dataset_eval=test_dataset)
    result = {"model": [model_name], "dataset": [dataset_train], "accuracy": [accuracy]}
    
    if dist.get_rank() == 0:
        save_results(OUTPUT_PATH, OUTPUT_NAME, pd.DataFrame(result))


if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)
