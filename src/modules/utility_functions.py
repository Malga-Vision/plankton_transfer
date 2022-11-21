from .constants import CROP_RATIO, IMAGENET_MEAN, IMAGENET_STD

from typing import Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

import timm
from timm.data import resolve_data_config

from imagedataset import (AdvancedImageFolder, 
                          LoaderPIL, 
                          ImageLoader, 
                          Interpolation, 
                          LoaderTurboJPEG)

from .backbone import Backbone
from .composed import ComposedModel
from dataset import get_dataset


def get_composed_model(backbone_name: str,
                       bottleneck_dim: int,
                       n_classes: int,
                       classifier_bias: Optional[bool] = True,
                       classifier_wn: Optional[bool] = False,
                       pretrained: Optional[bool] = True,
                       color_jitter_train: Optional[float] = 0.,
                       random_vflip: Optional[bool] = True,
                       random_hflip: Optional[bool] = True,
                       drop_rate: Optional[float] = None,
                       drop_path_rate: Optional[float] = None) -> ComposedModel:

    backbone = get_backbone(model_name=backbone_name, 
                            pretrained=pretrained, 
                            color_jitter_train=color_jitter_train,
                            random_vflip=random_vflip,
                            random_hflip=random_hflip,
                            drop_rate=drop_rate,
                            drop_path_rate=drop_path_rate)
                            
    bottleneck = get_bottleneck(backbone=backbone, bottleneck_size=bottleneck_dim)

    classifier_in_features = bottleneck_dim if bottleneck_dim > 0 else get_output_dim(backbone)
    classifier = nn.Linear(in_features=classifier_in_features, 
                           out_features=n_classes, 
                           bias=classifier_bias)

    if classifier_wn:
        classifier = weight_norm(classifier)

    return ComposedModel(backbone=backbone, classifier=classifier, bottleneck=bottleneck)


def get_torchvision_backbone(model_name: str,
                             pretrained: Optional[bool] = True,
                             color_jitter_train: Optional[float] = 0.,
                             jit_compile_transforms: Optional[bool] = False) -> Backbone:

    
    model = None

    getmodel_fnc = getattr(torchvision.models, model_name)
    model = getmodel_fnc(pretrained=pretrained)
    model_list = list(model.children())
    model = nn.Sequential(*model_list[:-1], nn.Flatten())

    train_transform = _get_transform(model, 
                                     random_crop=True, 
                                     color_jitter=color_jitter_train, 
                                     random_hflip=True,
                                     normalize=True)

    eval_transform = _get_transform(model, 
                                    random_crop=False, 
                                    color_jitter=0., 
                                    random_hflip=False,
                                    normalize=True)
    
    return Backbone(backbone=model,
                    name=model_name,
                    train_transform=train_transform, 
                    eval_transform=eval_transform,
                    jit_compile_transforms=jit_compile_transforms)



def load_model(model_name: str, checkpoint: str):

    backbone = get_torchvision_backbone(model_name)

    checkpoint = torch.load(checkpoint, map_location="cpu")["model"]
    del checkpoint["fc.weight"]
    del checkpoint["fc.bias"]

    model= torchvision.models.__dict__[model_name](num_classes=1, pretrained=False)
    model.load_state_dict(checkpoint, strict=False)

    model = list(model.children())[:-1] + [nn.Flatten()]
    model = torch.nn.Sequential(*model)
    backbone.backbone = model

    return backbone


def get_backbone(model_name: str,
                 pretrained: Optional[bool] = True,
                 color_jitter_train: Optional[float] = 0.,
                 random_vflip: Optional[bool] = True,
                 random_hflip: Optional[bool] = True,
                 drop_rate: Optional[float] = None,
                 drop_path_rate: Optional[float] = None) -> Backbone:
    """ 
        Create a backbone from the model name (as str).
        The model is retrieved from timm or from torchvision (if from_torcvision is True)

        Args:
            model_name (str): The name of the architecture.
            pretrained (bool, optioanl): True if imagenet weights should be used.
        Returns:
            the backbone

        Raises:
            ValueError if the model_name is not a timm model.
            CreateModelError if there is an error while creating model.
    """
    

    if model_name == "resnet50_whoi":
        return load_model("resnet50", "pretrain/resnet50_whoi.pth")
    elif model_name == "resnet50_kaggle":
        return load_model("resnet50", "pretrain/resnet50_kaggle.pth")
    elif model_name == "resnet50_zooscan":
        return load_model("resnet50", "pretrain/resnet50_zooscan.pth")
    elif model_name == "resnet50_in21k":
        return load_model("resnet50", "pretrain/resnet50_in21k.pth")
    if model_name == "resnet50":
        return get_torchvision_backbone(model_name="resnet50")

    model = None

    if model_name not in timm.list_models():
        raise ValueError(f"Model {model_name} not available in TIMM!")

    try:
        # num_classes=0 means not classifier, just feature extractor
        # (keeping last pooling).
        model = timm.create_model(model_name,
                                  pretrained=pretrained, 
                                  num_classes=0, 
                                  drop_rate=drop_rate,
                                  drop_path_rate=drop_path_rate)

    except Exception:
        raise CreateModelError("Error while crating the model!")

    train_transform = _get_transform(model, 
                                     random_crop=True, 
                                     color_jitter=color_jitter_train, 
                                     random_hflip=random_hflip,
                                     random_vflip=random_vflip,
                                     normalize=True)

    eval_transform = _get_transform(model, 
                                    random_crop=False, 
                                    color_jitter=0., 
                                    random_hflip=False,
                                    normalize=True)
    
    return Backbone(backbone=model,
                    name=model_name,
                    train_transform=train_transform, 
                    eval_transform=eval_transform,
                    jit_compile_transforms=False)


def get_bottleneck(backbone: Backbone, bottleneck_size: int) -> nn.Module:

    if bottleneck_size <= 0:
        return None

    bottleneck_in = get_output_dim(backbone)
    bottleneck_out = bottleneck_size

    normalization = nn.BatchNorm1d if _has_layer(backbone, nn.BatchNorm2d) else \
                    nn.LayerNorm
    activation = nn.ReLU if _has_layer(backbone, nn.ReLU) else nn.GELU

    return nn.Sequential(nn.Linear(bottleneck_in, bottleneck_out),
                          normalization(bottleneck_out),
                          activation())


def _has_layer(module: nn.Module, type) -> bool:

    children = list(module.children())

    if isinstance(module, type):
        return True
    
    output = False

    for child in children:
        output = output or _has_layer(child, type)

    return output

def _get_interpolation(model: Union[ComposedModel, Backbone, nn.Module],
                      use_pil: Optional[bool] = False,
                      from_torchvision: Optional[bool] = False) -> Interpolation:

    """
        Given a model returns the interpolation used on that model.

        Args:
            model (ComposedModel | Backbone | nn.Module): the model.
            use_pil (bool, optional): True to get PIL interpolations, False to get
            open-cv interpolations.
            from_torchvision (bool, optional): True if the model is from torchvision, 
            false if it is from timm.

        Returns:
            the interpolation.
    """

    # torchvision
    if from_torchvision:
        if use_pil:
            return Interpolation.PIL_BILINEAR
        else:
            return Interpolation.CV_BILINEAR

    # timm
    if isinstance(model, ComposedModel):
        model = model.backbone.backbone
    elif isinstance(model, Backbone):
        model = model.backbone

    config = resolve_data_config({}, model=model)

    interpolation_name = config["interpolation"]

    if interpolation_name == "bicubic":
        if use_pil:
            return Interpolation.PIL_BICUBIC
        else:
            return Interpolation.CV_BICUBIC
    else:
        if use_pil:
            return Interpolation.PIL_BILINEAR
        else:
            return Interpolation.CV_BILINEAR

 
def _get_imageloader(model: Union[ComposedModel, Backbone, nn.Module],
                     use_pil: Optional[bool] = True,
                     from_torchvision: Optional[bool] = False) -> ImageLoader:

    """
        Given a model returns the ImageLoader for that model.

        Args:
            model (ComposedModel | Backbone | nn.Module): the model.
            use_pil (bool, optional): True to get PIL interpolations, False to get
            open-cv interpolations.
            from_torchvision (bool, optional): True if the model is from torchvision, 
            false if it is from timm.

        Returns:
            the ImageLoader
    """
    interpolation = _get_interpolation(model, use_pil, from_torchvision)
    size = _get_loadersize(model)

    if use_pil:
        return LoaderPIL(size=(size,size), interpolation=interpolation)
    else:
        return LoaderTurboJPEG(size=(size,size), interpolation=interpolation)

      
        
def get_dataset_for_model(model: Union[ComposedModel, Backbone, nn.Module, str],
                          dataset_name: str,
                          use_pil: Optional[bool] = True,
                          from_database: Optional[bool] = False) -> AdvancedImageFolder:
    """
        Get the dataset with the correct input size for a given model.

        Args:
            model (nn.Module | Backbone | ComposedModel | str): the model or the name of
            the model.
            dataset_name (str): the name of the dataset to get.
            from_pil (bool, optional): do we need to use LoaderPIL?
            from_database (bool, optional): True to get the dataset from database.

        Returns:
            the AdvancedImageFolder
    """
    if isinstance(model, str):
        model = get_backbone(model, pretrained=False)

    loader = _get_imageloader(model, use_pil=use_pil)

    return get_dataset(dataset_name=dataset_name, 
                       from_database=from_database, 
                       percentage=None, 
                       loader=loader)

# public wrapper for _get_loader_size
def get_input_size(model: Union[ComposedModel, Backbone, nn.Module]) -> int:
    return _get_loadersize(model)


def _get_loadersize(model: Union[ComposedModel, Backbone, nn.Module]) -> int:
    """ 
        NOTE: given a model returns the loading `size` for images before any crops. 
        In particular the loader size is equal to model_input_size/crop_ratio.

        Example:
                 - loading raw image             ORIGINAL_SIZE (512, 640, 3)
              -> - resize(loader_size=256)       LOAD_SIZE     (256, 256, 3)
                 - crop(input_size=224)          CROP_SIZE     (224, 224, 3)
    

        Args:
            model (ComposedModel | Backbone | nn.Module): the model.
        Returns:
            the loadersize (int)
    """

    return int(_get_cropsize(model)/CROP_RATIO)


def _get_cropsize(model: Union[ComposedModel, Backbone, nn.Module]) -> int:

    """
        NOTE: given a model returns the model crop size. 

        Example:
                 - loading raw image             ORIGINAL_SIZE (512, 640, 3)
                 - resize(loader_size=256)       LOAD_SIZE     (256, 256, 3)
              -> - crop(input_size=224)          CROP_SIZE     (224, 224, 3)

        Args:
            model (ComposedModel | Backbone | nn.Module): the model.
            
        Returns:
            the cropsize (int) of the model (for example 224).
    """

    if isinstance(model, ComposedModel):
        model = model.backbone.backbone
    elif isinstance(model, Backbone):
        model = model.backbone

    config = resolve_data_config({}, model=model)

    return config["input_size"][1]



def _get_transform(model: Union[ComposedModel, Backbone, nn.Module],
                   random_crop: Optional[bool] = False,
                   random_hflip: Optional[bool] = False,
                   random_vflip: Optional[bool] = False,
                   color_jitter: Optional[float] = 0.,
                   normalize: Optional[bool] = True,
                   image_mean: Optional[Tuple] = None,
                   image_std: Optional[Tuple] = None) -> nn.Module:
    """
        Args:
            TODO
        Returns:
            a nn.Module (the transform).
    """
    
    if isinstance(model, ComposedModel):
        model = model.backbone.backbone
    elif isinstance(model, Backbone):
        model = model.backbone

    size = _get_cropsize(model)

    transforms = []

    if random_crop:
        transforms.append(T.RandomCrop(size))
    else:
        transforms.append(T.CenterCrop(size))

    if random_hflip:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    
    if random_vflip:
        transforms.append(T.RandomVerticalFlip(p=0.5))

    if color_jitter > 0:
        transforms.append(T.ColorJitter(brightness=color_jitter, 
                                        contrast  =color_jitter, 
                                        saturation=color_jitter))

    config = resolve_data_config({}, model=model)

    if image_mean is None:
        image_mean = config['mean']

    if image_std is None:
        image_std = config['std']

    if normalize:
        transforms.append(T.Normalize(mean=image_mean, std=image_std, inplace=True))

    return nn.Sequential(*transforms)
    


def get_output_dim(model: Union[Backbone, nn.Module, ComposedModel],
                   device: Optional[torch.device] = None,
                   n_channels: Optional[int] = 3) -> int:
    """
        Get the feature dimension of a module, given the input dimension of the images.

        Args:
            model (torch.nn.module | Backbone | ComposedModel): the module in 
            cosideration.
            device (torch.device): the device of the module.
            n_channels (int): the number of input channels.

        Returns:
            the output dimension.

        Raises:
            WrongOutputError if the model has a wrong output format.
    """
    if device is None:
        device = list(model.parameters())[0].device

    input_size = _get_loadersize(model)
    
    try:
        sample = torch.randn(1, n_channels, input_size, input_size)
        
        out = model(sample)

        # case ComposedModel
        if isinstance(out, tuple):
            out = out[1]
        output_dim = out.shape[1]

    except Exception:
        raise WrongOutputError("Wrong output!")

    return output_dim



# Error classes
class CreateModelError(Exception):
    """ Raised when there is an error while creating the model! """
    pass


class WrongOutputError(Exception):
    """ Raised when a model has a wrong output format! """
    pass
