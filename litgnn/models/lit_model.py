from typing import Dict

from torch import Tensor
import pytorch_lightning as L
from torch_geometric.data import Batch
from hydra.utils import instantiate as hydra_instantiate


class LitGNNModel(L.LightningModule):
    
    def __init__(
        self,
        model_config,
        train_config,
        task_config
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.task_config = task_config
        self.train_config = train_config
        self.model = hydra_instantiate(model_config, _convert_="all")
        self.loss_func = hydra_instantiate(self.task_config.loss)
        self._step_to_metrics = {
            step: {k: hydra_instantiate(v).to(self.device) for k, v in self.task_config.metrics.items()}
            for step in ("train", "val", "test")
        }

    def configure_optimizers(self) -> Dict:
        
        optimizer = hydra_instantiate(
            self.train_config.optimizer, 
            params=self.parameters(), 
            _recursive_=False, 
            _convert_="all"
        )
        lr_scheduler = hydra_instantiate(self.train_config.scheduler, optimizer=optimizer, _convert_="all")
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
        metric_to_func = self._step_to_metrics[step]
        for func in metric_to_func.values():
            func(preds, target) # Accumulate metrics
        return loss
    
    def _log_and_reset_metrics(self, step: str, **log_kwargs) -> None:

        for metric, func in self._step_to_metrics[step].items():
            # Compute metric on all batches
            value = func.compute().item()
            self.log(f"{step}_{metric}", value, **log_kwargs)
            # Reset internal state such that metric is ready for new data
            func.reset()