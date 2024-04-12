import logging
import itertools
from pathlib import Path
from functools import partial
from typing import Dict, List

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import MoleculeNet

from litgnn import splits as dataset_splits
from litgnn.data.custom_dataset import CustomDataset
from litgnn.models.cmpnn.featurization import atom_features, bond_features

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger()


def load_dataset(dataset_config) -> Dataset:

    dataset_type = dataset_config.dataset_type
    if dataset_type == "custom":
        kwargs = {k: v for k, v in dataset_config.items() if v}
        atom_messages = kwargs.pop("atom_messages", False)
        dataset = CustomDataset(
            root=str(Path.cwd() / ".cache"), 
            create_graph_from_smiles_fn=partial(create_mol_graph_from_smiles, atom_messages=atom_messages),
            **kwargs
        )
    elif dataset_type == "molecule_net":
        dataset = MoleculeNet(
            root=str(Path.cwd() / ".cache"), 
            name=dataset_config.dataset_name,
            pre_filter=lambda data: Chem.MolFromSmiles(data.smiles) is not None,
            pre_transform=lambda data: create_mol_graph_from_smiles(
                smiles=data.smiles, 
                y=data.y, 
                atom_messages=dataset_config.atom_messages
            )
        )
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


def create_mol_graph_from_smiles(smiles: str, **kwargs) -> Data:

    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices, edge_attrs = [], []
    for src, dst in itertools.combinations(range(x.size(0)), 2):
        if (bond := mol.GetBondBetweenAtoms(src, dst)) is None:
            continue
        e = bond_features(bond)
        edge_indices += [[src, dst], [dst, src]]
        edge_attrs += [e, e] if kwargs.get("atom_messages") else [x[src].tolist() + e, x[dst].tolist() + e]
    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        num_nodes=x.size(0),
        y=kwargs.get("y")
    )