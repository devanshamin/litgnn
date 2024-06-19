import logging
import random
from typing import Dict, Tuple

from torch_geometric.data import Dataset

logger = logging.getLogger()


def random_split(
    dataset: Dataset,
    split_sizes: Tuple[int] = (0.8, 0.1, 0.1),
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, Dataset]:

    random.seed(seed)
    train_size, val_size, _ = (int(sz * len(dataset)) for sz in split_sizes)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    splits = dict(
        train=dataset.index_select(train_indices),
        val=dataset.index_select(val_indices),
        test=dataset.index_select(test_indices)
    )

    if verbose:
        logger.info(
            f'\nTotal samples = {len(dataset):,} |',
            " | ".join(f'{s.capitalize()} set = {len(d):,}' for s, d in splits.items())
        )

    return splits
