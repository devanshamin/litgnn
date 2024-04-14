import copy
import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import torch
import numpy as np
import hydra
import wandb
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate as hydra_instantiate
import pytorch_lightning as L
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from litgnn.data.data_module import LitDataModule
from litgnn.nn.models.lit_model import LitGNNModel
from litgnn.utils import profile_execution, pre_init_model_setup

load_dotenv()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def prepare_config(cfg: DictConfig, data_module: LitDataModule) -> DictConfig:

    # Populate the required config values
    cfg.train.dataset.num_node_features = data_module.num_node_features
    cfg.train.dataset.num_edge_features = data_module.num_edge_features
    cfg.train.scheduler.steps_per_epoch = data_module.num_train_steps_per_epoch
    if cfg.dataset.dataset_type == "custom":
        # Add group key as the `wandb` job name prefix instead of 'custom'
        cfg.train.trainer.logger.job_type = f"{cfg.dataset.group_key}-{cfg.dataset.dataset_name}"

    # Resolve all variable interpolations
    OmegaConf.resolve(cfg)
    # logger.info(OmegaConf.to_yaml(cfg))

    return cfg


@profile_execution
def run(cfg) -> Tuple[float, Dict[str, float]]:

    seed = cfg.train.seed
    seed_everything(seed, workers=True)
    get_metrics = lambda callback_metrics: {
        k: v.item() for k, v in callback_metrics.items() if not k.endswith("loss")
    }
    
    data_module = LitDataModule(
        dataset_config=cfg.dataset, 
        split=cfg.train.dataset.split,
        split_sizes=cfg.train.dataset.split_sizes,
        batch_size=cfg.train.batch_size
    )
    cfg = prepare_config(cfg, data_module)
    pre_init_model_setup(model_config=cfg.model, data_module=data_module)
    lit_model = LitGNNModel(cfg)

    assert cfg.train.trainer.accelerator == "gpu" and torch.cuda.is_available(), \
        "GPU acceleration is requested but CUDA is not available."
    trainer: L.Trainer = hydra_instantiate(cfg.train.trainer, _recursive_=True, _convert_="all")
    if isinstance(trainer.logger, WandbLogger):
        for cb in trainer._callback_connector.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                # If you want to save model checkpoints in the `wandb` dir
                # ckpt_dir = Path(trainer.logger.experiment.dir, "checkpoints")
                run_id = trainer.logger.experiment.id
                ckpt_dir = Path.cwd() / f"lightning_logs/{run_id}"
                ckpt_dir.mkdir(parents=True)
                cb.dirpath = ckpt_dir

    trainer.fit(lit_model, datamodule=data_module)
    best_val_loss = trainer.callback_metrics["val_loss"].item()
    metrics = get_metrics(trainer.callback_metrics)

    trainer.test(lit_model, datamodule=data_module) # Automatically loads the best weights
    metrics.update(get_metrics(trainer.callback_metrics))

    wandb.finish()

    return best_val_loss, metrics 


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    hydra_config = HydraConfig.get()
    sweeper = hydra_config.runtime.choices.get("hydra/sweeper")
    if (hydra_config.mode == RunMode.MULTIRUN) and (sweeper == "optuna"):
        # For HPO, a single run is enough to obtain the val_loss
        # Ignore `cfg.train.num_seed_runs`
        best_val_loss, _ = run(cfg)
        return best_val_loss # For hydra optuna sweep
    
    metrics = dict(train={}, val={}, test={})
    start_seed = cfg.train.seed
    for i in range(start_seed, start_seed + cfg.train.num_seed_runs):
        _cfg = copy.deepcopy(cfg)
        _cfg.train.seed = i
        _, test_metrics = run(_cfg)
        for k, v in test_metrics.items():
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