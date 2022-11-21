
import argparse
from tabnanny import verbose
from utils import load_config
from termcolor import colored
import os
from os.path import join, isfile, basename
from pathlib import Path
import torch
from torch.nn import Softmax
from sklearn.metrics import f1_score
import pandas as pd

from modules import (BasicFineTuner, get_backbone, get_dataset_for_model, 
                     extract_features, get_composed_model)
import torch.distributed as dist
dist.init_process_group("nccl")


CONFIG_ARCHITECTURE = "config/architecture.yaml"

parser = argparse.ArgumentParser(description='Eval f1 score of model')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ckpt_load',  type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--out', type=str)


def main(model_name, dataset_name, ckpt_load, batch_size, out):

    torch.backends.cudnn.benchmark = False
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    # get hparams
    hparams_architecture = load_config(CONFIG_ARCHITECTURE)

    # extract hparams for architecture
    color_jitter    = hparams_architecture["color_jitter"]
    random_hflip    = hparams_architecture["random_hflip"]
    random_vflip    = hparams_architecture["random_vflip"]
    drop_rate       = hparams_architecture["drop_rate"]
    drop_path_rate  = hparams_architecture["drop_path_rate"]

    bottleneck_dim  = hparams_architecture["bottleneck_dim"]
    classifier_bias = hparams_architecture["classifier_bias"]
    classifier_wn   = hparams_architecture["classifier_wn"]
    
   
    # get dataset 
    dataset = get_dataset_for_model(model_name, dataset_name)
    n_classes = dataset.n_classes()

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
                               drop_path_rate=drop_path_rate).to(local_rank)


    ckpt_load = join(ckpt_load)
    if verbose: print(f"Loading ckpt from: {ckpt_load}")

    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    state_dict = torch.load(ckpt_load, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)

    _, labels, logits = extract_features(model=model, 
                                                datasets=dataset, 
                                                batch_size=batch_size,
                                                verbose=False,
                                                return_logits=True)
    labels = labels.cpu()    
    _, cp = torch.max(logits, dim=1)
    cp = cp.cpu().numpy()
    micro = f1_score(labels, cp, average="micro")
    macro = f1_score(labels, cp, average="macro")

    current = pd.DataFrame({"model":[basename(ckpt_load)], "dataset": [dataset_name], "micro":[micro], "macro": [macro]})

    if dist.get_rank() == 0:
        if isfile(out):
            df = pd.read_csv(out)
            df = pd.concat([df, current], axis=0)
            df.to_csv(out, index=False)
        else:
            current.to_csv(out, index=False)

        print(f"Dataset: {dataset_name}")
        print(f"Model: {basename(ckpt_load)}")
        print(f"micro {micro}")
        print(f"macro {macro}")
        print()
           


if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)
