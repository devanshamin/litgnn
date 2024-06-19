import inspect
import sys
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import requests
from pydantic import BaseModel, computed_field, field_validator
from torch_geometric.data import download_url
from tqdm import tqdm


def get_available_groups() -> Dict[str, "CustomDatasetSpec"]:

    available_groups = {}
    for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if name.endswith("DatasetSpec") and hasattr(cls, "group_key"):
            available_groups[cls.group_key] = cls
    return available_groups


class CustomDatasetSpec(BaseModel):
    group_key: str # Used to identify group of datasets coming from an org i.e., Biogen, TDC etc.
    dataset_name: str
    smiles_col_idx: int
    target_col_idx: Union[int, List[int]]
    url: Optional[str] = None
    file_path: Optional[str] = None
    file_name: Optional[Union[str, List[str]]] = None

    @field_validator("url", "file_path", mode="after", check_fields=True)
    def check_url_file_path(cls, v, values):

        if not v and not values.get("file_path"):
            raise ValueError("Both 'url' and 'file_path' cannot be empty.")
        return v

    @computed_field
    @property
    def display_name(self) -> str:

        return self.dataset_name.upper()


class BiogenDatasetSpec(CustomDatasetSpec):
    """Biogen ADME datasets. Futher information can be found \
    [here](https://devanshamin.netlify.app/posts/molecule-property-prediction-datasets/).

    Following are the individual tasks within the dataset:
    - `HLM`: Human Liver Microsomal stability
    - `MDR1_ER`: MDR1-MDCK Efflux Ratio
    - `Sol`: Aqueous Solubility
    - `hPPB`: Human Plasma Protein Binding
    - `rPPB`: Rat Plasma Protein Binding
    - `RLM`: Rat Liver Microsomal stability
    """

    group_key: ClassVar[str] = "biogen"
    smiles_col_idx: int = 0
    target_col_idx: int = -1
    url: str = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/MPNN/ADME_{}"
    file_name: List[str] = ["train_val.csv", "test.csv"]

    datasets: ClassVar[Tuple[str]] = ("HLM", "MDR1_ER", "Sol", "hPPB", "rPPB", "RLM")

    @field_validator("dataset_name", mode="after")
    def check_dataset_name(cls, v, values):

        if v not in cls.datasets:
            raise ValueError(f"Invalid dataset name! Please select one from {cls.datasets}")
        return v

    @computed_field
    @property
    def display_name(self) -> str:

        return self.dataset_name

    def download_data(self, dir_path: str) -> None:

        # The dataset will be saved as follows,
        # - biogen -> HLM -> train_val.csv
        # - biogen -> HLM -> test.csv
        # where, HLM is the dataset name
        dir_path = Path(dir_path, self.display_name)
        for i, fname in enumerate(self.file_name):
            # They use train/test (https://github.com/molecularinformatics/Computational-ADME/tree/main/MPNN)
            suffix = "train" if fname.startswith("train_val") else "test"
            url = self.url.format(f"{self.display_name}_{suffix}.csv")
            download_url(url, dir_path, filename=fname)
            # Update file name
            self.file_name[i] = str(Path(self.display_name, fname))


class TDCADMETDatasetSpec(CustomDatasetSpec):
    """[Therapeutic Data Commons (TDC)](https://tdcommons.ai) AMDET group benchmark datasets."""

    group_key: ClassVar[str] = "tdc"
    smiles_col_idx: int = 1
    target_col_idx: int = -1
    # URL is created from TDC GitHub repo,
    # Base URL - https://github.com/mims-harvard/TDC/blob/905bd6fb8fd92b25ce3c80dddeba562f95a135a2/tdc/utils/load.py#L249
    # ID - https://github.com/mims-harvard/TDC/blob/905bd6fb8fd92b25ce3c80dddeba562f95a135a2/tdc/metadata.py#L888
    url: str = "https://dataverse.harvard.edu/api/access/datafile/4426004"
    # Each dataset has it's own directory with `train_val.csv` and `test.csv` files.
    file_name: List[str] = ["train_val.csv", "test.csv"]

    # List of datasets under the ADMET group
    datasets: ClassVar[Tuple[str]] = (
        "ames",
        "bbb_martins",
        "bioavailability_ma",
        "caco2_wang",
        "clearance_hepatocyte_az",
        "clearance_microsome_az",
        "cyp2c9_substrate_carbonmangels",
        "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels",
        "cyp2d6_veith",
        "cyp3a4_substrate_carbonmangels",
        "cyp3a4_veith",
        "dili",
        "half_life_obach",
        "herg",
        "hia_hou",
        "ld50_zhu",
        "lipophilicity_astrazeneca",
        "pgp_broccatelli",
        "ppbr_az",
        "solubility_aqsoldb",
        "vdss_lombardo"
    )

    @field_validator("dataset_name", mode="after")
    def check_dataset_name(cls, v, values):

        if v not in cls.datasets:
            raise ValueError(f"Invalid dataset name! Please select one from {cls.datasets}")
        return v

    def download_data(self, dir_path: str) -> None:

        # Adapted from https://github.com/mims-harvard/TDC/blob/905bd6fb8fd92b25ce3c80dddeba562f95a135a2/tdc/utils/load.py#L159C1-L180C25
        zip_save_path = Path(dir_path, "admet_group.zip")
        unzipped_path = Path(dir_path, zip_save_path.stem)
        if not unzipped_path.exists():
            response = requests.get(self.url, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
            with open(zip_save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            with ZipFile(zip_save_path, "r") as zip:
                zip.extractall(path=unzipped_path)
            zip_save_path.unlink()

        # Update file name
        # After unzip, the directory structure is as follows,
        # - admet_group/admet_group/$DATASET/train_val.csv
        # - admet_group/admet_group/$DATASET/test.csv
        self.file_name = [str(Path(unzipped_path.name, unzipped_path.name, self.dataset_name, nm)) for nm in self.file_name]
