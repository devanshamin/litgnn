import warnings
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
import numpy as np
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate as hydra_instantiate
from torchmetrics.metric import Metric
from pytorch_lightning import seed_everything

from litgnn.data.utils import get_dataloaders

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def train(dataloader, model, optimizer, scheduler, loss_fn):

    loss_sum = 0.0
    for batch in tqdm(dataloader, desc="Training", total=len(dataloader), leave=False):
        model.zero_grad()
        batch = batch.to(device)
        preds = model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        loss = loss_fn(preds, batch.y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += loss.item()
    return loss_sum / len(dataloader)


@torch.inference_mode()
def evaluate(dataloader, model, loss_fn, metric_fns: Optional[Dict[str, Metric]] = None):

    loss_sum = 0.0
    metrics = {}

    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader), leave=False):
        batch = batch.to(device)
        preds = model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        loss = loss_fn(preds, batch.y)
        if metric_fns is not None:
            for fn in metric_fns.values():
                fn(preds, batch.y) # Accumulate metrics
        loss_sum += loss.item() 
    
    if metric_fns is not None:
        for metric, fn in metric_fns.items():
            # Compute metric on all batches
            metrics[metric] = fn.compute().item()
            # Reset internal state such that metric ready for new data
            fn.reset()

    return loss_sum / len(dataloader), metrics


def _run(cfg):

    seed = cfg.train.seed
    seed_everything(seed, workers=True)
    dataloaders = get_dataloaders(cfg)
    OmegaConf.resolve(cfg)
    logger.info(OmegaConf.to_yaml(cfg))
    
    model = hydra_instantiate(cfg.model).to(device)
    optimizer = hydra_instantiate(cfg.train.optimizer, params=model.parameters(), _recursive_=False)
    scheduler = hydra_instantiate(
        cfg.train.scheduler,
        optimizer=optimizer,
        steps_per_epoch=len(dataloaders["train"]),
    )
    loss_fn = hydra_instantiate(cfg.task.loss_fn)
    metric_fns = {k: hydra_instantiate(v).to(device) for k, v in cfg.task.metrics.items()}
    get_metrics_str = lambda prefix, metrics: " | ".join(f"{prefix} {k}={v:.3f}" for k, v in metrics.items())
    
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"No. of model parameters: {num_params:,}")
    logger.info(model)

    best_model, best_val_loss = None, torch.inf
    er_counter = 0
    for epoch in range(cfg.train.epochs):
        train_loss = train(dataloaders["train"], model, optimizer, scheduler, loss_fn)
        val_loss, metrics = evaluate(dataloaders["val"], model, loss_fn, metric_fns)
        logger.info(
            f"Epoch {epoch:<3} | Train loss={train_loss:.3f} | Valid loss={val_loss:.3f} | "
            + get_metrics_str("Valid", metrics)
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            er_counter = 0
        else:
            er_counter += 1
        if er_counter == cfg.train.early_stopping:
            break

    test_loss, test_metrics = evaluate(dataloaders["test"], best_model, loss_fn, metric_fns)
    logger.info(f"[Seed {seed}] Test loss={test_loss:.3f} | " + get_metrics_str("Test", test_metrics))

    output_dir = Path(HydraConfig.get().runtime.output_dir) / str(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_obj = dict(
        cfg=cfg, 
        state_dict=model.state_dict(), 
        best_val_loss=best_val_loss, 
        test_loss=test_loss,
        test_metrics=test_metrics
    )
    torch.save(save_obj, Path(output_dir, "model.pt"))

    return best_val_loss, test_metrics 


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> Optional[float]:
    
    hydra_config = HydraConfig.get()
    sweeper = hydra_config.runtime.choices.get("hydra/sweeper")
    if (hydra_config.mode == RunMode.MULTIRUN) and (sweeper == "optuna"):
        # For HPO, a single run is enough to obtain the val_loss
        # Ignore `cfg.train.num_seed_runs`
        best_val_loss, _ = _run(cfg)
        return best_val_loss # For hydra optuna sweep
    
    metrics = {}
    start_seed = cfg.train.seed
    for i in range(start_seed, start_seed + cfg.train.num_seed_runs):
        cfg.train.seed = i
        _, test_metrics = _run(cfg)
        for k, v in test_metrics.items():
            metrics.setdefault(k, []).append(v)
    
    for k, v in metrics.items():
        logger.info("Test {}: {} ± {}".format(k, round(np.mean(v), 3), round(np.std(v), 3)))


if __name__ == "__main__":
    
    run()