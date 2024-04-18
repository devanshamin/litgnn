import logging
from pathlib import Path
from typing import Dict, List

from rdkit import Chem, RDLogger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Dataset
from torch_geometric.datasets import MoleculeNet
from hydra.utils import instantiate as hydra_instantiate

from litgnn import splits as dataset_splits
from litgnn.data.custom_dataset import CustomDataset

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger()


def load_dataset(dataset_config: DictConfig) -> Dataset:

    cfg = OmegaConf.to_container(dataset_config)
    cfg["root"] = cfg.pop("save_dir", str(Path.cwd() / ".cache"))
    if pre_transform := cfg.pop("pre_transform", None):
        cfg["pre_transform"] = hydra_instantiate(pre_transform, _convert_="all")

    kwargs = dict(
        root=cfg["root"],
        name=cfg["dataset_name"],
        transform=cfg.get("transform"),
        pre_transform=cfg.get("pre_transform"),
        pre_filter=lambda data: Chem.MolFromSmiles(data.smiles) is not None,
        force_reload=cfg.get("force_reload", False),
    )
    dataset_type = cfg["dataset_type"]
    if dataset_type == "custom":
        dataset = CustomDataset(group_key=cfg["group_key"], **kwargs)
    elif dataset_type == "molecule_net":
        dataset = MoleculeNet(**kwargs)
    else:
        raise ValueError(f"Invalid dataset type '{dataset_type}'!")
    return dataset


def get_dataset_splits(
    dataset: Dataset, 
    *, 
    split: str, 
    split_sizes: List[float],
    verbose: bool = False
) -> Dict[str, Dataset]:

    split_func = getattr(dataset_splits, split, None)
    assert split_func is not None, f"Invalid split '{split}'!"
    if hasattr(dataset, "train_test_split_idx") and (split_idx := dataset.train_test_split_idx):
        train_val_dataset = dataset.index_select(split_idx["train_val"])
        splits = split_func(train_val_dataset, split_sizes=split_sizes, balanced=True, verbose=0)
        splits["val"] += splits["test"]
        splits["test"] = dataset.index_select(split_idx["test"])
        if verbose:
            logger.info(
                f'Total samples = {len(train_val_dataset) + len(split_idx["test"]):,} | '
                + " | ".join(f'{s.capitalize()} set = {len(d):,}' for s, d in splits.items())
            )
    else:
        splits = split_func(dataset, split_sizes=split_sizes, balanced=True, verbose=verbose)

    return splits
