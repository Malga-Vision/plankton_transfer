import argparse
from threading import local
from utils import load_config
from termcolor import colored

from os.path import join
from pathlib import Path

from modules import BasicFineTuner, get_composed_model, get_dataset_for_model
import torch.distributed as dist
dist.init_process_group("nccl")

CKPT_FOLDER = "outputs/cache/"
CONFIG_ARCHITECTURE = "config/architecture.yaml"
CONFIG_TRAINING = "config/training.yaml"


parser = argparse.ArgumentParser(description='EVALParser')
parser.add_argument('--dataset',     type=str)
parser.add_argument('--ckpt',        type=str)
parser.add_argument('--model_name',  type=str)
parser.add_argument('--from_torchvision', action='store_true')


def main(model_name, dataset, ckpt, from_torchvision):


    ckpt_load = join(CKPT_FOLDER, ckpt)

    # get hparams
    hparams_architecture = load_config(CONFIG_ARCHITECTURE)
    hparams_training     = load_config(CONFIG_TRAINING)

    # extract hparams for architecture
    color_jitter    = hparams_architecture["color_jitter"]
    bottleneck_dim  = hparams_architecture["bottleneck_dim"]
    classifier_bias = hparams_architecture["classifier_bias"]
    classifier_wn   = hparams_architecture["classifier_wn"]
    
    
    # get dataset 
    dataset = get_dataset_for_model(model_name, dataset)
    n_classes = dataset.n_classes()

    # get model  
    model = get_composed_model(backbone_name=model_name, 
                               from_torchvision=from_torchvision, 
                               pretrained=True, 
                               color_jitter_train=color_jitter, 
                               jit_compile_transforms=False,
                               n_classes=n_classes, 
                               bottleneck_dim=bottleneck_dim, 
                               classifier_bias=classifier_bias, 
                               classifier_wn=classifier_wn)


    finetuner = BasicFineTuner(model=model,
                               ckpt_name=ckpt,
                               config_dict=hparams_training)

    finetuner.load(ckpt_load, load_classifier=True)
    accuracy = finetuner.eval(dataset)
    
    if dist.get_rank() == 0:
        print(f"Accuracy: {accuracy}")

if __name__=='__main__':
    args = vars(parser.parse_args())
    main(**args)
