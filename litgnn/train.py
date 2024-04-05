import warnings
import logging
from pathlib import Path
from typing import Optional, Dict

import torch
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate as hydra_instantiate
from torchmetrics.metric import Metric

from litgnn.utils import NoamLR
from litgnn.data.utils import get_dataloaders
from litgnn.models.graph_level import GraphLevelGNN

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def train(dataloader, model, optimizer, scheduler, loss_fn):

    loss_sum = 0.0
    for batch in tqdm(dataloader, desc="Training", total=len(dataloader), leave=False):
        model.zero_grad()
        batch = batch.to(DEVICE)
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
        batch = batch.to(DEVICE)
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
            metrics[metric] = fn.compute()
            # Reset internal state such that metric ready for new data
            fn.reset()

    return loss_sum / len(dataloader), metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> float:
    
    dataloaders = get_dataloaders(cfg)
    OmegaConf.resolve(cfg)
    logger.info(OmegaConf.to_yaml(cfg))

    model = GraphLevelGNN(**cfg.model).to(DEVICE)
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(f"No. of model parameters: {num_params:,}")
    logger.info(model)
    optimizer = hydra_instantiate(cfg.train.optimizer, params=model.parameters(), _recursive_=False)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[cfg.train.scheduler.warmup_epochs],
        total_epochs=[cfg.train.epochs],
        steps_per_epoch=len(dataloaders["train"]),
        init_lr=[cfg.train.scheduler.init_lr],
        max_lr=[cfg.train.scheduler.max_lr],
        final_lr=[cfg.train.scheduler.final_lr]
    )
    loss_fn = hydra_instantiate(cfg.task.loss_fn)
    metric_fns = {k: hydra_instantiate(v) for k, v in cfg.task.metrics.items()}
    get_metrics_str = lambda prefix, metrics: " | ".join(f"{prefix} {k}={v:.3f}" for k, v in metrics.items())
    
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

    test_loss, metrics = evaluate(dataloaders["test"], best_model, loss_fn, metric_fns)
    logger.info(f"Test loss={test_loss:.3f} | " + get_metrics_str("Test", metrics))

    output_dir = HydraConfig.get().runtime.output_dir
    save_obj = dict(
        cfg=cfg, 
        state_dict=model.state_dict(), 
        best_val_loss=best_val_loss, 
        test_loss=test_loss,
        metrics=metrics
    )
    torch.save(save_obj, Path(output_dir, "model.pt"))

    return best_val_loss # For hydra optuna sweep


if __name__ == "__main__":
    
    run()