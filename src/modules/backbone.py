from typing import Optional
import torch.nn as nn
import torch

class Backbone(nn.Module):

    def __init__(self, 
                 backbone: nn.Module, 
                 train_transform: Optional[nn.Module] = None, 
                 eval_transform: Optional[nn.Module] = None,
                 jit_compile_transforms: Optional[bool] = False,
                 name: Optional[str] = None):
        """
            A backbone object is a model + the input transforms. 
            At training time the train_transform will be used, at eval time the eval 
            transform will be used.

            Args:
                backbone (nn.Module): the backbone.
                train_transform (nn.Module, optional): the transform for training.
                eval_transform (nn.Module, optional): the transform for validation and 
                inference.
                jit_compile_transforms(bool, optional): True to compile transforms with
                jit.
                name (str, optional): the name of the backbone.
        """

        super().__init__()
        self.backbone = backbone
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        
        if jit_compile_transforms:
            self.train_transform = torch.jit.script(self.train_transform)
            self.eval_transform = torch.jit.script(self.eval_transform)

        self.name = name if name is not None else "unnamed"


    def __repr__(self):
        return f"<Backbone '{self.name}'>"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Takes a Tensor and returns the output of the model. """

        if self.training and (self.train_transform is not None):
            x = self.train_transform(x)
        elif (not self.training) and (self.eval_transform is not None):
            x = self.eval_transform(x)

        return self.backbone(x)