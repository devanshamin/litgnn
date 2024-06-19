import logging
from pathlib import Path
from typing import Optional

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate as hydra_instantiate
from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import SuccessiveHalvingPruner
from optuna.trial import TrialState

import optuna
from litgnn.trainer.optuna import OptunaObjective

logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


def get_default_storage(cfg: DictConfig) -> str:

    optuna_dir = Path.cwd() / "optuna"
    optuna_dir.mkdir(exist_ok=True)
    fname = "{}_{}.db".format(cfg.model.model_cls, cfg.dataset.dataset_name)
    return "sqlite:///" + str(optuna_dir / fname.lower())


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:

    # Resolve all variable interpolations
    OmegaConf.resolve(cfg)

    hydra_config = HydraConfig.get()
    sweeper = hydra_config.sweeper
    optuna_sweeper: OptunaSweeper = hydra_instantiate(sweeper)

    objective = OptunaObjective(cfg, optuna_sweeper)
    storage = sweeper.storage or get_default_storage(cfg)
    sampler = hydra_instantiate(sweeper.sampler)
    directions = optuna_sweeper.sweeper._get_directions()
    study = optuna.create_study(
        study_name=sweeper.study_name,
        storage=storage,
        sampler=sampler,
        directions=directions,
        load_if_exists=True,
        pruner=SuccessiveHalvingPruner()
    )
    logger.info(f"Study name: {study.study_name}")
    logger.info(f"Storage: {storage}")
    logger.info(f"Sampler: {type(sampler).__name__}")
    logger.info(f"Monitor: {cfg.optuna_lightning_pruning_callback.monitor}")
    logger.info(f"Directions: {directions}")

    study.optimize(objective, n_trials=sweeper.n_trials, timeout=sweeper.get("timeout"), show_progress_bar=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    logger.info(f"No. of finished trials: {len(study.trials)}")
    logger.info(f"No. of pruned trials: {len(pruned_trials)}")
    logger.info(f"No. of complete trials: {len(complete_trials)}")
    trial = study.best_trial
    params = tuple(f"{k!s}={v}" for k, v in trial.params.items())
    best_trial = dict(trial_no=trial.number, value=trial.value, params=params)
    if run_id := trial.user_attrs.get("run_id"):
        best_trial["run_id"] = run_id
        best_trial["ckpt_dir"] = trial.user_attrs.get("ckpt_dir")
    logger.info("Best trial:")
    logger.info(OmegaConf.to_yaml(DictConfig(best_trial)))


if __name__ == "__main__":

    main()
