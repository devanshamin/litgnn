import hashlib
import os
import os.path as osp
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_gz,
                                  extract_zip)
from torch_geometric.utils import from_smiles

from litgnn.data.dataset_spec import CustomDatasetSpec, get_available_groups


class CustomDataset(InMemoryDataset):
    """A custom PyG dataset collection consisting of public datasets."""

    def __init__(
        self,
        root: str,
        name: str,
        group_key: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:

        groups = get_available_groups()
        dataset_spec_cls = groups.get(group_key)
        assert dataset_spec_cls is not None, f"Invalid group key! Please select one from {list(groups)}."
        self.dataset_spec: CustomDatasetSpec = dataset_spec_cls(dataset_name=name)
        self._split_idx = None # Used when the dataset provides separate files for training and testing

        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)

        # Load saved dataset
        self.load(self.processed_paths[0])
        if len(self.processed_paths) > 1:
            self._split_idx = torch.load(self.processed_paths[1])

    @property
    def dataset_name(self) -> str:

        return self.dataset_spec.dataset_name

    @property
    def train_test_split_idx(self) -> Optional[Dict[str, range]]:

        return self._split_idx

    @property
    def raw_dir(self) -> str:

        return osp.join(self.root, self.dataset_spec.group_key, "raw")

    @property
    def processed_dir(self) -> str:

        return osp.join(self.root, self.dataset_spec.group_key, "processed", self.dataset_name)

    @property
    def raw_file_names(self) -> Union[str, List[str]]:

        return self.dataset_spec.file_name

    @property
    def processed_file_names(self) -> Union[str, List[str]]:

        fname = "data.pt"
        if isinstance(self.dataset_spec.file_name, list):
            # Assumes the dataset provides separate files for train and test
            fname = [fname, "idx.pt"]
        return fname

    def download(self) -> None:

        if CustomDataset._has_download_method(self.dataset_spec, "download_data"):
            self.dataset_spec.download_data(self.raw_dir)
        else:
            path = download_url(self.dataset_spec.url, self.raw_dir)
            if self.dataset_spec.file_name.endswith("gz"):
                extract_gz(path, self.raw_dir)
                os.unlink(path)
            elif self.dataset_spec.file_name.endswith("zip"):
                extract_zip(path, self.raw_dir)
                os.unlink(path)

    def _get_data_list(self, file_path):

        with open(file_path, "r") as f:
            dataset = f.read().split("\n")[1:-1]
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        data_list = []
        for line in dataset:
            line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
            values = line.split(",")

            smiles = values[self.dataset_spec.smiles_col_idx]
            labels = values[self.dataset_spec.target_col_idx]
            labels = labels if isinstance(labels, list) else [labels]

            ys = [float(y) if len(y) > 0 else float("NaN") for y in labels]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            data = from_smiles(smiles)
            data.y = y

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        return data_list

    def process(self) -> None:

        if len(self.raw_paths) > 1:
            split_idx = {}
            data_list = []
            start = end = 0
            for file_path in self.raw_paths:
                dl = self._get_data_list(file_path)
                data_list += dl
                end = len(dl)
                split_idx[Path(file_path).stem] = range(start, start + end)
                start = end
            assert len(self.processed_paths) > 1
            torch.save(split_idx, self.processed_paths[1])
        else:
            data_list = self._get_data_list(self.raw_paths[0])

        self.save(data_list, self.processed_paths[0])

    def __repr__(self) -> str:

        return f"{self.dataset_spec.display_name}({len(self)})"

    @staticmethod
    def _has_download_method(cls, method_name):

        return hasattr(cls, method_name) and callable(getattr(cls, method_name))

    @staticmethod
    def create_hash(input_str: str) -> str:

        hash_object = hashlib.sha256(input_str.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig
