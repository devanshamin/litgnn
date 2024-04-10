import os
from typing import List, Dict

import pytorch_lightning as L
from torch_geometric.loader import DataLoader

from litgnn.data.utils import load_dataset, get_dataset_splits


class LitDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        dataset_config: Dict,
        split: str,
        split_sizes: List[float],
        batch_size: int
    ) -> None:
        
        super().__init__()
        self.dataset_config = dataset_config
        self.split = split
        self.split_sizes = split_sizes
        self.batch_size = batch_size

        self._dataset = None
        self._splits = None

    @property
    def num_node_features(self) -> int:
        
        if self._dataset is None:
            self.setup(stage="fit")
        return self._dataset.num_node_features
    
    @property
    def num_edge_features(self) -> int:
        
        if self._dataset is None:
            self.setup(stage="fit")
        return self._dataset.num_edge_features

    @property
    def num_train_steps_per_epoch(self) -> int:
        
        if self._splits is None:
            self.setup(stage="fit")
        return len(self.train_dataloader())

    def setup(self, stage: str) -> None:
        
        self._dataset = load_dataset(self.dataset_config)
        self._splits = get_dataset_splits(self._dataset, split=self.split, split_sizes=self.split_sizes)

    def train_dataloader(self) -> DataLoader:
        
        return DataLoader(
            self._splits["train"], 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True,
            # num_workers=os.cpu_count(), # Causes issues with hydra joblib launcher 
        )

    def val_dataloader(self) -> DataLoader:
        
        return DataLoader(
            self._splits["val"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            # num_workers=os.cpu_count(),
        )

    def test_dataloader(self) -> DataLoader:
        
        return DataLoader(
            self._splits["test"], 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            # num_workers=os.cpu_count(),
        )