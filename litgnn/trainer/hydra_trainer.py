import logging
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate as hydra_instantiate
import pytorch_lightning as L
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from litgnn.data.data_module import LitDataModule
from litgnn.nn.models.lit_model import LitGNNModel
from litgnn.trainer.utils import profile_execution, pre_init_model_setup

load_dotenv()
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class RunOutput:
    val_loss: float
    metrics: Dict[str, float]
    run_id: Optional[str] = None
    ckpt_dir: Optional[str] = None


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
def hydra_trainer(cfg: DictConfig, callbacks: Optional[List[Any]] = None) -> RunOutput:

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
    if callbacks is not None:
        trainer._callback_connector.trainer.callbacks.extend(callbacks)
    run_id = ckpt_dir = None
    if isinstance(trainer.logger, WandbLogger):
        for cb in trainer._callback_connector.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                # If you want to save model checkpoints in the `wandb` dir
                # ckpt_dir = Path(trainer.logger.experiment.dir, "checkpoints")
                run_id = trainer.logger.experiment.id
                ckpt_dir = Path.cwd() / f"lightning_logs/{run_id}"
                # If a optuna trial gets pruned, it will reuse the same dir for the next trial
                # hence exist_ok is set to True.
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                cb.dirpath = ckpt_dir

    trainer.fit(lit_model, datamodule=data_module)
    best_val_loss = trainer.callback_metrics["val_loss"].item()
    metrics = get_metrics(trainer.callback_metrics)

    trainer.test(lit_model, datamodule=data_module) # Automatically loads the best weights
    metrics.update(get_metrics(trainer.callback_metrics))

    if run_id:
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Checkpoint dir: {str(ckpt_dir)}")

    wandb.finish()

    return RunOutput(val_loss=best_val_loss, metrics=metrics, run_id=run_id, ckpt_dir=str(ckpt_dir))
