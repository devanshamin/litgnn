from typing import Dict

from torch import Tensor
import pytorch_lightning as L
from torch_geometric.data import Batch
from omegaconf import DictConfig
from hydra.utils import instantiate as hydra_instantiate


class LitGNNModel(L.LightningModule):
    
    def __init__(self, config: DictConfig) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = hydra_instantiate(config.model, _convert_="all")
        self.loss_func = hydra_instantiate(config.dataset.task.loss)
        for step in ("train", "val", "test"):
            for metric, v in config.dataset.task.metrics.items():
                setattr(self, f"{step}_{metric}", hydra_instantiate(v))

    def configure_optimizers(self) -> Dict:
        
        optimizer = hydra_instantiate(
            self.config.train.optimizer, 
            params=self.parameters(), 
            _recursive_=False, 
            _convert_="all"
        )
        lr_scheduler = hydra_instantiate(self.config.train.scheduler, optimizer=optimizer, _convert_="all")
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def forward(self, batch: Batch) -> Tensor:

        return self.model(
            x=batch.x, 
            edge_index=batch.edge_index, 
            edge_attr=batch.edge_attr,
            batch=batch.batch
        )

    def training_step(self, batch, batch_idx: int) -> Tensor:
        
        preds = self(batch)
        return self._calculate_loss_and_metrics(preds, target=batch.y, step="train")

    def on_train_epoch_end(self) -> None:

        self._log_and_reset_metrics(step="train", logger=True)

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        
        preds = self(batch)
        return self._calculate_loss_and_metrics(preds, target=batch.y, step="val")

    def on_validation_epoch_end(self) -> None:

        self._log_and_reset_metrics(step="val", logger=True)

    def test_step(self, batch, batch_idx: int) -> Tensor:
        
        preds = self(batch)
        return self._calculate_loss_and_metrics(preds, target=batch.y, step="test")
    
    def on_test_epoch_end(self) -> None:

        self._log_and_reset_metrics(step="test", logger=True)
    
    def _calculate_loss_and_metrics(
        self, 
        preds: Tensor, 
        target: Tensor, 
        step: str,
    ) -> Tensor:

        loss = self.loss_func(preds, target)
        self.log(
            f"{step}_loss", 
            loss.item(), 
            on_step=False, 
            on_epoch=True, 
            logger=True, 
            prog_bar=step in ("train", "val"),
            batch_size=preds.size(0)
        )
        if self.config.dataset.task.task_type.endswith("classification"):
            target = target.long()
        for metric in self.config.dataset.task.metrics:
            func = getattr(self, f"{step}_{metric}")
            func(preds, target) # Accumulate metrics
        return loss
    
    def _log_and_reset_metrics(self, step: str, **log_kwargs) -> None:

        for metric in self.config.dataset.task.metrics:
            func = getattr(self, f"{step}_{metric}")
            # Compute metric on all batches
            value = func.compute().item()
            self.log(f"{step}_{metric}", value, **log_kwargs)
            # Reset internal state such that metric is ready for new data
            func.reset()