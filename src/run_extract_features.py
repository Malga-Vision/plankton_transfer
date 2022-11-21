
import argparse
from tabnanny import verbose
from utils import load_config
from termcolor import colored
import os
from os.path import join
from pathlib import Path
from modules import (BasicFineTuner, get_backbone, get_dataset_for_model, 
                     extract_features, get_composed_model)

import torch
import torch.distributed as dist
dist.init_process_group("nccl")

OUTPUT_DEFAULT = "outputs/features/"

parser = argparse.ArgumentParser(description='Feature extractor Parser')
parser.add_argument('--datasets',   nargs='+', type=str)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ckpt_load',  type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--out', type=str)


def main(model_name, datasets, ckpt_load, batch_size, out):

    if out is None:
        out = OUTPUT_DEFAULT
    
    print(f"OUT: {out}")
    torch.backends.cudnn.benchmark = False

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    Path(out).mkdir(exist_ok=True)


    model = get_composed_model(backbone_name=model_name, 
                               pretrained=True, 
                               color_jitter_train=0, 
                               random_hflip=False,
                               random_vflip=False,
                               n_classes=100, #RANDOM NUMBER, IT WILL NOT BE USED...
                               bottleneck_dim=512, 
                               classifier_bias=True, 
                               classifier_wn=True,
                               drop_rate=0.,
                               drop_path_rate=0.).to(local_rank)

    if ckpt_load:
        ckpt_load = join(ckpt_load)

        if verbose: print(f"Loading ckpt from: {ckpt_load}")

        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        state_dict = torch.load(ckpt_load, map_location=map_location)

        for k in list(state_dict.keys()):
            if "classifier" in k:
                 print(k)
                 del state_dict[k]
                 
        backbone.load_state_dict(state_dict, strict=False)


    for dataset in datasets:
        current_dataset = get_dataset_for_model(backbone, dataset)
        features, labels = extract_features(model=backbone, 
                                            datasets=current_dataset, 
                                            batch_size=batch_size,
                                            verbose=False,
                                            return_logits=False)
        
        features = features.cpu()
        labels = labels.cpu()
        
        if rank == 0:
            torch.save(features, join(out, f"{model_name}_{dataset}_features"))
            torch.save(labels, join(out, f"{model_name}_{dataset}_labels"))
           
        dist.barrier()

if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)
