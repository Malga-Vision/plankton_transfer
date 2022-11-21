from typing import Optional, Tuple, List

import os

import torch
import torch.nn as nn
from .backbone import Backbone


class ComposedModel(nn.Module):

    def __init__(self, 
                 backbone: Backbone, 
                 classifier: Optional[nn.Module] = None, 
                 bottleneck: Optional[nn.Module] = None):
        """
            Compose a backbone, a bottleneck (optional) and a classifier into a single
            model.

            Args:
                backbone   (Backbone): the backbone.
                classifier (nn.Module): the classifier
                bottleneck (nn.Module): an optional bottleneck between backbone and 
                classifier.
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.bottleneck = bottleneck


    def get_param_groups(self, 
                         lr_backbone: float, 
                         lr_classifier: Optional[float] = None, 
                         lr_bottleneck: Optional[float] = None) -> List[dict]:
        """
            Returns the param group list setting different learning rates for the 
            backbone, classifier and bottleneck.

            Args:
                lr_backbone (float): the learning rate for the backbone.
                lr_classifier (float or None): the learning rate for the classifier, 
                if None the lr of the bottleneck will be used, if neither lr_classifier
                nor lr_bottleneck are set, the lr of the backbone is used.
                lr_bottleneck (float or None): the learning rate for the bottlenck.
                if None the lr of the classifier will be used

            Returns:
                a list with the param groups.
        """

        if lr_bottleneck is None:
            if lr_classifier is not None:
                # set to lr_classifier
                lr_bottleneck = lr_classifier
            else:
                # neither lr_bottleneck nor lr_classifier are specified
                lr_bottleneck = lr_backbone
        
        if lr_classifier is None:
            lr_classifier = lr_bottleneck
        
        param_group = [{"params": self.backbone.parameters(), "lr": lr_backbone}]

        if self.bottleneck is not None:
            param_group.append({"params": self.bottleneck.parameters(), 
                                "lr": lr_bottleneck})
    
        if self.classifier is not None:
            param_group.append({"params": self.classifier.parameters(), 
                                "lr": lr_classifier})

        if lr_backbone <= 0:
            self.freeze_backbone()
        
        if self.bottleneck and lr_bottleneck <= 0:
            self.freeze_bottleneck()

        if self.classifier and lr_classifier <= 0:
            self.freeze_classifier()


        return param_group


    def forward(self, x: torch.Tensor) -> Tuple:

        """
            Forward method, it outputs also the features before and after the 
            bottleneck.

            Args:
                x (torch.Tensor): input tesnor
            
            Returns:
                a triple: (backbone_out, bottleneck_out, classifier_out)
        """

        features_backbone = self.backbone(x)

        features_bottleneck = features_backbone

        if self.bottleneck is not None:
            features_bottleneck = self.bottleneck(features_bottleneck)
        
        if self.classifier is not None:
            out = self.classifier(features_bottleneck)
            return features_backbone, features_bottleneck, out
        
        return features_backbone, features_bottleneck



    def save(self, *path: str):
        """ Saves weights to path. """
        path = os.path.join(*path)
        torch.save(self.state_dict(), path)


    def load(self, *path: str):
        """ Loads weights from path. """
        path = os.path.join(*path)
        self.load_state_dict(torch.load(path))


    def freeze_classifier(self):
        """ Freezes the classifier. """
        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = False


    def freeze_bottleneck(self):
        """ Freezes the bottleneck. """
        if self.bottleneck is not None:
            for p in self.bottleneck.parameters():
                p.requires_grad = False


    def freeze_backbone(self):
        """ Freezes the backbone. """
        for p in self.backbone.parameters():
            p.requires_grad = False
