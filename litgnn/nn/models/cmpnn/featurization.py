import itertools
from typing import List, Union

import torch
from rdkit import Chem
from torch_geometric.data import Data


# Adapted from https://github.com/SY575/CMPNN/blob/master/chemprop/features/featurization.py
class FeaturesGenerator:

    def __init__(self, atom_messages: bool = False, max_atomic_num: int = 100) -> None:
        self.atom_messages = atom_messages
        self.atom_features = {
            "atomic_num": list(range(max_atomic_num)),
            "degree": [0, 1, 2, 3, 4, 5],
            "formal_charge": [-1, -2, 1, 2, 0],
            "chiral_tag": [0, 1, 2, 3],
            "num_Hs": [0, 1, 2, 3, 4],
            "hybridization": [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
            # IsAromatic
            # Mass
        }
    
    @property
    def atom_fdim(self) -> int:
        return sum(len(choices) + 1 for choices in self.atom_features.values()) + 2

    @property
    def bond_fdim(self) -> int:
        return 14 if self.atom_messages else 147

    @staticmethod
    def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
        """Creates a one-hot encoding.
        
        Args:
            value: The value for which the encoding should be one.
            choices: A list of possible values.
        Returns:
            A one-hot encoding of the value in a list of length len(choices) + 1. If value is 
            not in the list of choices, then the final element in the encoding is 1.
        """
        encoding = [0] * (len(choices) + 1)
        index = choices.index(value) if value in choices else -1
        encoding[index] = 1

        return encoding

    def generate_atom_features(self, atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
        """Builds a feature vector for an atom.

        Args:
            atom: An RDKit atom.
            functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
        Returns:
            A list containing the atom features.
        """
        features = (
            FeaturesGenerator.onek_encoding_unk(atom.GetAtomicNum() - 1, self.atom_features["atomic_num"])
            + FeaturesGenerator.onek_encoding_unk(atom.GetTotalDegree(), self.atom_features["degree"])
            + FeaturesGenerator.onek_encoding_unk(atom.GetFormalCharge(), self.atom_features["formal_charge"])
            + FeaturesGenerator.onek_encoding_unk(int(atom.GetChiralTag()), self.atom_features["chiral_tag"])
            + FeaturesGenerator.onek_encoding_unk(int(atom.GetTotalNumHs()), self.atom_features["num_Hs"])
            + FeaturesGenerator.onek_encoding_unk(
                int(atom.GetHybridization()), self.atom_features["hybridization"]
            )
            + [1 if atom.GetIsAromatic() else 0]
            + [atom.GetMass() * 0.01]  # Scaled to about the same range as other features
        )
        if functional_groups is not None:
            features += functional_groups
        return features


    def generate_bond_features(self, bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
        """Builds a feature vector for a bond.

        Args:
            bond: A RDKit bond.
        Returns:
            A list containing the bond features.
        """
        if bond is None:
            bond_fdim = 14
            fbond = [1] + [0] * (bond_fdim - 1)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == Chem.rdchem.BondType.SINGLE,
                bt == Chem.rdchem.BondType.DOUBLE,
                bt == Chem.rdchem.BondType.TRIPLE,
                bt == Chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)
            ]
            fbond += FeaturesGenerator.onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond

    def __call__(self, data: Data) -> Data:
        
        mol = Chem.MolFromSmiles(data.smiles)
        x = torch.tensor([self.generate_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
        edge_indices, edge_attrs = [], []
        for src, dst in itertools.combinations(range(x.size(0)), 2):
            if (bond := mol.GetBondBetweenAtoms(src, dst)) is None:
                continue
            e = self.generate_bond_features(bond)
            edge_indices += [[src, dst], [dst, src]]
            edge_attrs += [e, e] if self.atom_messages else [x[src].tolist() + e, x[dst].tolist() + e]
        edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data
