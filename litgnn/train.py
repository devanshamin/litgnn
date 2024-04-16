import copy
import logging
from typing import Optional

import torch
import numpy as np
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode

from litgnn.trainer import hydra_trainer

load_dotenv()
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    hydra_config = HydraConfig.get()
    sweeper = hydra_config.runtime.choices.get("hydra/sweeper")
    if (hydra_config.mode == RunMode.MULTIRUN) and (sweeper == "optuna"):
        # For HPO, a single run is enough to obtain the val_loss
        # Ignore `cfg.train.num_seed_runs`
        out = hydra_trainer(cfg)
        return out.val_loss # For hydra optuna sweep
    
    metrics = dict(train={}, val={}, test={})
    start_seed = cfg.train.seed
    for i in range(start_seed, start_seed + cfg.train.num_seed_runs):
        _cfg = copy.deepcopy(cfg)
        _cfg.train.seed = i
        out = hydra_trainer(_cfg)
        for k, v in out.metrics.items():
            split = k.split("_")[0]
            metrics[split].setdefault(k, []).append(v)
    
    for result in metrics.values():
        for metric, value in result.items():
            logger.info(
                "{}: {} ± {}".format(
                    metric.capitalize(), 
                    round(np.mean(value), 3), 
                    round(np.std(value), 3)
                )
            )


if __name__ == "__main__":
    
    main()