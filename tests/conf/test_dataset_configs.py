import pytest
from omegaconf import OmegaConf
from hydra import compose, initialize_config_module


BIOGEN_DATASETS = [
    ("biogen", "hlm", {"dataset_name": "HLM", "num_classes": 1, "group_key": "biogen", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("biogen", "rlm", {"dataset_name": "RLM", "num_classes": 1, "group_key": "biogen", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("biogen", "hppb", {"dataset_name": "hPPB", "num_classes": 1, "group_key": "biogen", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("biogen", "rppb", {"dataset_name": "rPPB", "num_classes": 1, "group_key": "biogen", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("biogen", "sol", {"dataset_name": "Sol", "num_classes": 1, "group_key": "biogen", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("biogen", "mdr1_er", {"dataset_name": "MDR1_ER", "num_classes": 1, "group_key": "biogen", "dataset_type": "custom", "task": {"task_type": "regression"}}),
]

TDC_DATASETS = [
    ("tdc", "ames", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "binary_classification"}}),
    ("tdc", "bbb_martins", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "binary_classification"}}),
    ("tdc", "cyp2c9_veith", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "binary_classification"}}),
    ("tdc", "cyp2d6_veith", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "binary_classification"}}),
    ("tdc", "cyp3a4_veith", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "binary_classification"}}),
    ("tdc", "ld50_zhu", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("tdc", "lipophilicity_astrazeneca", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "regression"}}),
    ("tdc", "solubility_aqsoldb", {"num_classes": 1, "group_key": "tdc", "dataset_type": "custom", "task": {"task_type": "regression"}}),
]

MOLECULE_NET_DATASETS = [
    ("molecule_net", "clintox", {"num_classes": 2, "dataset_type": "molecule_net", "task": {"task_type": "multilabel_classification"}}),
]


@pytest.mark.parametrize(
    "dir_name, dataset_name, expected_config",
    (
        *BIOGEN_DATASETS, *TDC_DATASETS, *MOLECULE_NET_DATASETS
    )
)
def test_dataset_config_excluding_metrics(dir_name, dataset_name, expected_config) -> None:

    with initialize_config_module(version_base=None, config_module="litgnn.conf"):
        cfg = compose(config_name="config", overrides=[f"dataset={dir_name}/{dataset_name}", "model=pna"])
        dataset_cfg = OmegaConf.to_container(cfg["dataset"])
        dataset_cfg["task"].pop("metrics")
        dataset_cfg.pop("save_dir", None)
        dataset_cfg.pop("pre_transform", None)
        
        # Build the expected config
        if "dataset_name" not in expected_config:
            expected_config["dataset_name"] = dataset_name
        task_config = expected_config["task"]
        if task_config["task_type"] in ("binary_classification", "multilabel_classification"):
            loss = "BCEWithLogitsLoss" 
        elif task_config["task_type"] == "multiclass_classification":
            loss = "CrossEntropyLoss"
        elif task_config["task_type"] == "regression":
            loss = "MSELoss"
        task_config["loss"] = {"_target_": f"torch.nn.{loss}"} 

        assert dataset_cfg == expected_config