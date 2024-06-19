import logging
import time
from typing import Callable

from omegaconf import DictConfig

from litgnn.data.data_module import LitDataModule
from litgnn.nn import models

logger = logging.getLogger()


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
