import sys
import inspect
from pathlib import Path
from zipfile import ZipFile
from typing import Dict, Optional, Union, List, Tuple, ClassVar

import requests
from tqdm import tqdm
from pydantic import BaseModel, field_validator, computed_field


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
    - `HLM_CLint`: Human Liver Microsomal stability
    - `MDR1-MDCK_ER`: MDR1-MDCK Efflux Ratio
    - `SOLUBILITY`: Aqueous Solubility
    - `hPPB`: Human Plasma Protein Binding
    - `rPPB`: Rat Plasma Protein Binding
    - `RLM_CLint`: Rat Liver Microsomal stability
    """

    group_key: ClassVar[str] = "biogen"
    dataset_name: str = "biogen_adme"
    smiles_col_idx: int = 2
    url: str = "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"
    file_name: str = "ADME_public_set_3521.csv"

    # Single CSV file containing separate columns for each task
    # Following are the columns in the dataset respresenting individual tasks
    column_to_idx: ClassVar[Dict[str, int]] = {
        "LOG HLM_CLint (mL/min/kg)": 4,
        "LOG MDR1-MDCK ER (B-A/A-B)": 5,
        "LOG SOLUBILITY PH 6.8 (ug/mL)": 6,
        "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)": 7,
        "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)": 8,
        "LOG RLM_CLint (mL/min/kg)": 9,
    }

    @field_validator("target_col_idx", mode="after")
    def check_target_col_idx(cls, v, values):
        target_col_idxs = cls.column_to_idx.values()
        if v not in target_col_idxs:
            raise ValueError(f"Invalid target column index! Please select one from {list(target_col_idxs)}")
        return v

    @computed_field
    @property
    def display_name(self) -> str:
        col_idx_to_name = {
            4: "HLM_CLint", 
            5: "MDR1-MDCK_ER",
            6: "SOLUBILITY",
            7: "hPPB",
            8: "rPPB",
            9: "RLM_CLint",
        }
        return col_idx_to_name[self.target_col_idx]


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

    @field_validator("target_col_idx", mode="after")
    def check_dataset_name(cls, v, values):
        if values.get("dataset_name") not in cls.datasets:
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