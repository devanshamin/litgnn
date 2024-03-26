import os
import random
from typing import Dict, Set, Tuple
from multiprocessing import Pool

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Dataset


def _get_scaffold(smiles: str, include_chirality: bool) -> str:
    
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=Chem.MolFromSmiles(smiles), 
        includeChirality=include_chirality
    )


def get_scaffold_to_indices(dataset: Dataset, include_chirality: bool = False) -> Dict[str, Set[int]]:

    scaffolds = {}
    with Pool(os.cpu_count()) as pool:
        iterable = ((ex.smiles, include_chirality) for ex in dataset)
        for i, scaffold in enumerate(pool.starmap(_get_scaffold, iterable)):
            scaffolds.setdefault(scaffold, set()).add(i)
    return scaffolds


def scaffold_split(
    dataset: Dataset, 
    split_sizes: Tuple[int] = (0.8, 0.1, 0.1),
    seed: int = 42, 
    balanced: bool = False,
    verbose: bool = False
) -> Dict[str, Dataset]:
    
    train_size, val_size, test_size = (int(sz * len(dataset)) for sz in split_sizes)
    scaffold_to_indices = get_scaffold_to_indices(dataset)

    index_sets = list(scaffold_to_indices.values())
    if balanced:  
        # Put indexes that's bigger than half of the val/test size into train
        # and rest indexes randomly
        big_index_sets, small_index_sets = [], []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  
        # Sort from largest to smallest scaffold sets
        index_sets = sorted(index_sets, key=len, reverse=True)
    
    train_indices, val_indices, test_indices = [], [], []
    train_scaffold_count = val_scaffold_count = test_scaffold_count = 0
    for index_set in index_sets:
        if len(train_indices) + len(index_set) <= train_size:
            train_indices += index_set
            train_scaffold_count += 1
        elif len(val_indices) + len(index_set) <= val_size:
            val_indices += index_set
            val_scaffold_count += 1
        else:
            test_indices += index_set
            test_scaffold_count += 1
    splits = dict(
        train=dataset.index_select(train_indices),
        val=dataset.index_select(val_indices),
        test=dataset.index_select(test_indices)
    )
    
    if verbose:
        print(
            f'Total scaffolds = {len(scaffold_to_indices):,} | '
            f'Train scaffolds = {train_scaffold_count:,} | '
            f'Val scaffolds = {val_scaffold_count:,} | '
            f'Test scaffolds = {test_scaffold_count:,}'
            f'\nTotal samples = {len(dataset):,} |',
            " | ".join(f'{s.capitalize()} set = {len(d):,}' for s, d in splits.items())
        )
    
    return splits