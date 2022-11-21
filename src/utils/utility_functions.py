from typing import Any, Dict, List

import inspect
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from ruamel.yaml import YAML
from . import schedulers


def optimizer_by_name(name: str,
                      param_groups: List[Dict[str, Any]],
                      hparams_dict: Dict[str, Any],
                      **hparams) -> torch.optim.Optimizer:
    """
        Returns the optimizer by name.

        Args:
            name (str): the name of the optimizer.
            param_groups list[Dict]: the list of param groups for the optimizer.
            hparams_dict (Dict): hparams for the optimizer.
            **hparams: additional (optional) hparams for the optimizer. If one or more
            of these hparams are not real arguments for the optimizer, then they will not
            be used.

        Returns:
            the torch.optim.Optimizer
    """

    optimizer_cls = getattr(torch.optim, name)

    try:
        return optimizer_cls(param_groups, **(hparams | hparams_dict))
    except Exception:
        return optimizer_cls(param_groups, **hparams_dict)


def scheduler_by_name(name: str, 
                      optimizer: torch.optim.Optimizer, 
                      hparams_dict: Dict[str, Any], 
                      **hparams):
    """
        Returns the optimizer by name.

        Args:
            name (str): the name of the optimizer.
            optimizer (torch.optim.Optimizer): the optimizer.
            hparams_dict (Dict): hparams for the scheduler.
            **hparams: additional (optional) hparams for the scheduler. If one or more
            of these hparams are not real arguments for the scheduler, then they will not
            be used.

        Returns:
            the torch.optim.Optimizer
    """

    # get custom schedulers
    cust_schedulers = [m[0] for m in inspect.getmembers(schedulers, inspect.isclass)]

    # get scheduler module for the selected one.
    scheduler_module = (
        schedulers if name in cust_schedulers else torch.optim.lr_scheduler
    )

    # get scheduler class
    scheduler_cls = getattr(scheduler_module, name)
    
    try:
        return scheduler_cls(optimizer, **(hparams | hparams_dict))
    except Exception:
        return scheduler_cls(optimizer, **hparams_dict)


def load_config(*path):
    """
        Load an external config yaml file and returns a dict.

        Args:
            *path (strings): the path strings to the config file.
        Returns:
            A dict represneting the config yaml file.
    """

    with open(os.path.join(*path)) as f:
        config = dict(YAML().load(f))

    return config



