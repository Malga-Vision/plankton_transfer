from typing import Optional
from torch.optim import Optimizer


class ExponentialStepScheduler:
    
    def __init__(self, 
                 optimizer: Optimizer, 
                 num_steps: int, 
                 gamma: Optional[float] = 10, 
                 power: Optional[float] = 0.75):
        """
            Initialized the lr scheduler.
            At iteration i the decay will be calculated as:

            .. math::
                decay(i) = \left(1 + \frac{gamma \cdot i}{max\_iter} \right)^{-power}

            and the learning rate will be calculated (for each param group) as:

            .. math::
                lr(i) = lr(0) \cdot decay

            where lr(0) is the initial learning rate (for the param group).

            Args:
                optimizer (torch.Optimizer): the optimizer to be scheduled.
                num_steps (int): maximum number of iterations.
                gamma (float): the gamma value to calculate decay.
                power (float): the power value to calculate decay.
        """

        self.iter_num = 0
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.gamma = gamma
        self.power = power

        self._init_optimizer()


    def _init_optimizer(self):
        """
            Initializes the optimizer by copying lr to lr0 in each param group.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr0'] = param_group['lr']


    def step(self):
        """
            Updates the learning rate of each param group.
        """
        decay = (1 + self.gamma * self.iter_num / self.num_steps) ** (-self.power)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay

        self.iter_num += 1