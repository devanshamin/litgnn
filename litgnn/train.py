import warnings

import torch
import torch.nn as nn
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import AUROC

from litgnn.data.utils import get_dataloaders
from litgnn.models.graph_level import GraphLevelGNN

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader, model, optimizer, loss_func):

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
        loss = loss_func(preds, batch.y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / len(dataloader)

@torch.inference_mode()
def evaluate(dataloader, model, loss_func, callable_metric=None):

    loss_sum = 0.0
    metric = 0.0
    for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader), leave=False):
        batch = batch.to(DEVICE)
        preds = model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )
        loss = loss_func(preds, batch.y)
        if callable_metric is not None:
            metric += callable_metric(preds, batch.y).item()
        loss_sum += loss.item() 
    return loss_sum / len(dataloader), metric / len(dataloader)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig):
    
    dataloaders = get_dataloaders(cfg)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    model = GraphLevelGNN(**cfg.model).to(DEVICE)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, params=model.parameters(), _recursive_=False)
    loss_func = nn.BCEWithLogitsLoss()
    
    best_model, best_loss = None, torch.inf
    for epoch in range(cfg.train.epochs):
        train_loss = train(dataloaders["train"], model, optimizer, loss_func)
        val_loss, _ = evaluate(dataloaders["val"], model, loss_func)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
        print(f"Epoch {epoch} | Train loss={train_loss:.3f} | Valid loss={val_loss:.3f}")

    auroc = AUROC(task="binary")
    test_loss, metric = evaluate(dataloaders["test"], best_model, loss_func, callable_metric=auroc)
    name = auroc.__class__.__name__
    print(f"Test loss={test_loss:.3f} | {name}={metric:.3f}")

if __name__ == "__main__":
    
    run()