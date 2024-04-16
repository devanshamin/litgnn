import logging
from pathlib import Path
from typing import Optional

import torch
import optuna
from optuna.trial import TrialState
from optuna.pruners import SuccessiveHalvingPruner
from optuna.integration import PyTorchLightningPruningCallback
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate as hydra_instantiate
from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper
from hydra_plugins.hydra_optuna_sweeper._impl import create_params_from_overrides
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path

from litgnn.trainer import hydra_trainer

logger = logging.getLogger(__name__)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


class Objective:
    
    def __init__(self, cfg: DictConfig, optuna_sweeper: OptunaSweeper) -> None:
        
        self.cfg = cfg
        self.optuna_sweeper = optuna_sweeper

        params_conf = self.optuna_sweeper.sweeper._parse_sweeper_params_config()
        self._search_space_distributions, self._fixed_params = create_params_from_overrides(params_conf)
        # Remove fixed parameters from Optuna search space
        for param_name in self._fixed_params:
            if param_name in self._search_space_distributions:
                del self._search_space_distributions[param_name]

        search_path = create_config_search_path(search_path_dir=str(Path.cwd() / "litgnn/conf"))
        self._cfg_loader = ConfigLoaderImpl(search_path)

    def _update_cfg(self, trial: optuna.trial.Trial) -> DictConfig:
                
        # Select the hyperparameters for the trial
        overrides = self.optuna_sweeper.sweeper._configure_trials(
            [trial], 
            self._search_space_distributions, 
            self._fixed_params
        )[0]
        logger.info(f"Selected hparams: {overrides}")
        
        # Merge the user provided task arguments with the chosen hyperparameters
        master_config = HydraConfig.instance().cfg
        overrides = list(overrides) + OmegaConf.to_container(master_config.hydra.overrides.task)

        # Override the chosen hyperparameters to the master config 
        cfg = self._cfg_loader.load_sweep_config(master_config, list(overrides))
        return cfg

    def __call__(self, trial: optuna.trial.Trial) -> float:
        
        cfg = self._update_cfg(trial)
        monitor = cfg.optuna_lightning_pruning_callback.monitor
        callback = PyTorchLightningPruningCallback(trial=trial, monitor=monitor)
        out = hydra_trainer(cfg, callbacks=[callback])

        if out.run_id:
            trial.set_user_attr(key="run_id", value=out.run_id)
            trial.set_user_attr(key="ckpt_dir", value=out.ckpt_dir)

        if monitor == "val_loss":
            metric_value = out.val_loss
        else:
            for k, v in out.metrics.items():
                if k == monitor:
                    metric_value = v
                    break
        return metric_value


@hydra.main(version_base="1.3.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    
    # Resolve all variable interpolations
    OmegaConf.resolve(cfg)

    hydra_config = HydraConfig.get()
    sweeper = hydra_config.sweeper
    optuna_sweeper: OptunaSweeper = hydra_instantiate(sweeper)
    
    objective = Objective(cfg, optuna_sweeper)
    storage = sweeper.storage
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