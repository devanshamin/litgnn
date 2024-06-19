import shutil
from pathlib import Path

import pytest
from omegaconf import DictConfig

from litgnn.data.utils import load_dataset


@pytest.mark.parametrize(
    "config",
    (
        dict(dataset_type="custom", group_key="tdc", dataset_name="ames", pre_transform=dict(_target_="litgnn.nn.models.cmpnn.featurization.FeaturesGenerator", atom_messages=False)),
        dict(dataset_type="custom", group_key="tdc", dataset_name="bbb_martins", pre_transform=dict(_target_="litgnn.nn.models.cmpnn.featurization.FeaturesGenerator", atom_messages=True)),
        dict(dataset_type="custom", group_key="biogen", dataset_name="HLM", pre_transform=dict(_target_="litgnn.nn.models.cmpnn.featurization.FeaturesGenerator", atom_messages=True)),
        dict(dataset_type="custom", group_key="biogen", dataset_name="rPPB", pre_transform=dict(_target_="litgnn.nn.models.cmpnn.featurization.FeaturesGenerator", atom_messages=False)),
        dict(dataset_type="molecule_net", dataset_name="clintox", pre_transform=dict(_target_="litgnn.nn.models.cmpnn.featurization.FeaturesGenerator", atom_messages=True)),
    )
)
def test_atom_messages_flag(config) -> None:

    expected = 14 if config["pre_transform"]["atom_messages"] else 147
    tmp_dir = Path.cwd() / "tmp"
    config["save_dir"] = str(tmp_dir)
    dataset = load_dataset(DictConfig(config))
    shutil.rmtree(tmp_dir)

    assert dataset.num_edge_features == expected
