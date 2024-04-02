from functools import partial
from pathlib import Path
import shutil

import pytest
from rdkit import Chem, RDLogger
from torch_geometric.datasets import MoleculeNet

from litgnn.data.utils import create_mol_graph_from_smiles
from litgnn.data.custom_dataset import CustomDataset

RDLogger.DisableLog("rdApp.*")


@pytest.mark.parametrize(
    "dataset_type, group_key, dataset_name, target_col_idx, atom_messages", 
    (
        ("cust", "tdc", "bbb_martins", None, True),
        ("custom", "tdc", "bbb_martins", None, True), 
        ("custom", "biogen", None, 6, False), 
        ("molecule_net", None, "ESOL", None, True), 
    )
)
def test_atom_messages_flag(
    dataset_type: str, 
    group_key: str, 
    dataset_name: str, 
    target_col_idx: int,
    atom_messages: bool
) -> None:

    expected = 14 if atom_messages else 147
    tmp_dir = Path.cwd() / "tmp"
    
    if dataset_type == "custom":
        fnc = partial(create_mol_graph_from_smiles, atom_messages=atom_messages)
        kwargs = dict(root=str(tmp_dir), group_key=group_key, create_graph_from_smiles_fn=fnc)
        if dataset_name:
            kwargs["dataset_name"] = dataset_name
        if target_col_idx:
            kwargs["target_col_idx"] = target_col_idx
        dataset = CustomDataset(**kwargs)
    elif dataset_type == "molecule_net":
        dataset = MoleculeNet(
            root=str(tmp_dir), 
            name=dataset_name,
            pre_filter=lambda data: Chem.MolFromSmiles(data.smiles) is not None,
            pre_transform=lambda data: create_mol_graph_from_smiles(
                smiles=data.smiles, 
                y=data.y, 
                atom_messages=atom_messages
            )
        )
    else:
        raise UserWarning(f"Invalid dataset type '{dataset_type}'!")
        
    shutil.rmtree(tmp_dir)
    assert dataset.num_edge_features == expected
    