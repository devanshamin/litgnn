import itertools
from pathlib import Path
from functools import partial
from typing import Dict

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from litgnn import splits as dataset_splits
from litgnn.data.custom_dataset import CustomDataset
from litgnn.models.cmpnn.featurization import atom_features, bond_features

RDLogger.DisableLog("rdApp.*")


def get_dataloaders(cfg) -> Dict[str, DataLoader]:

    dataset_type = cfg.dataset.dataset_type
    if dataset_type == "custom":
        kwargs = {k: v for k, v in cfg.dataset.items() if v}
        atom_messages = kwargs.pop("atom_messages")
        dataset = CustomDataset(
            root=str(Path.cwd() / ".cache"), 
            create_graph_from_smiles_fn=partial(create_mol_graph_from_smiles, atom_messages=atom_messages),
            **kwargs
        )
    elif dataset_type == "molecule_net":
        dataset = MoleculeNet(
            root=str(Path.cwd() / ".cache"), 
            name=cfg.dataset.dataset_name,
            pre_filter=lambda data: Chem.MolFromSmiles(data.smiles) is not None,
            pre_transform=lambda data: create_mol_graph_from_smiles(
                smiles=data.smiles, 
                y=data.y, 
                atom_messages=cfg.dataset.atom_messages
            )
        )

    cfg.train.dataset.num_node_features = dataset.num_node_features
    cfg.train.dataset.num_edge_features = dataset.num_edge_features

    split_func = getattr(dataset_splits, cfg.train.dataset.split)
    split_sizes = cfg.train.dataset.split_sizes
    if hasattr(dataset, "train_test_split_idx") and (split_idx := dataset.train_test_split_idx):
        train_val_dataset = dataset.index_select(split_idx["train_val"])
        splits = split_func(train_val_dataset, split_sizes=split_sizes, balanced=True, verbose=0)
        splits["val"] += splits["test"]
        splits["test"] = dataset.index_select(split_idx["test"])
        print(
            f'\nTotal samples = {len(train_val_dataset) + len(split_idx["test"]):,} |',
            " | ".join(f'{s.capitalize()} set = {len(d):,}' for s, d in splits.items())
        )
    else:
        splits = split_func(dataset, split_sizes=split_sizes, balanced=True, verbose=1)
    
    dataloaders = {
        split: DataLoader(data, batch_size=cfg.train.batch_size, shuffle=split=="train", pin_memory=True)
        for split, data in splits.items()
    }
    return dataloaders


def create_mol_graph_from_smiles(smiles: str, **kwargs) -> Data:

    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
    edge_indices, edge_attrs = [], []
    for src, dst in itertools.combinations(range(x.size(0)), 2):
        if (bond := mol.GetBondBetweenAtoms(src, dst)) is None:
            continue
        e = bond_features(bond)
        edge_indices += [[src, dst], [dst, src]]
        edge_attrs += [e, e] if kwargs.get("atom_messages") else [x[src].tolist() + e, x[dst].tolist() + e]
    edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

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