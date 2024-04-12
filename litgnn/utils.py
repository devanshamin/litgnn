import logging
import time
from typing import List, Union, Callable

import numpy as np
from litgnn import models
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from litgnn.data.data_module import LitDataModule

logger = logging.getLogger()


class NoamLR(_LRScheduler):
    """Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    
    Args:
        optimizer: A PyTorch optimizer.
        warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        total_epochs: The total number of epochs.
        steps_per_epoch: The number of steps (batches) per epoch.
        init_lr: The initial learning rate.
        max_lr: The maximum learning rate (achieved after warmup_epochs).
        final_lr: The final learning rate (achieved after total_epochs).
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float]
    ) -> None:
        
        grps = (optimizer.param_groups, warmup_epochs, total_epochs, init_lr, max_lr, final_lr)
        assert len(set(map(len, grps))) == 1

        self.num_lrs = len(optimizer.param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        
        return list(self.lr)

    def step(self, current_step: int = None):
        """Updates the learning rate by taking a step."""

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def profile_execution(func: Callable) -> Callable:

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    
    return wrapper


def pre_init_model_setup(model_config: DictConfig, data_module: LitDataModule) -> None:

    model_cls = model_config.model_cls
    if model_cls == "PNA":
        if data_module._splits is None:
            data_module.setup()
        cls = getattr(models, model_cls)
        cls.compute_degree(dataloader=data_module.train_dataloader())

    return None