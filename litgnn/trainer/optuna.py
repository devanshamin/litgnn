import logging
from pathlib import Path

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_optuna_sweeper.optuna_sweeper import OptunaSweeper
from hydra_plugins.hydra_optuna_sweeper._impl import create_params_from_overrides
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.utils import create_config_search_path

from litgnn.trainer.hydra_trainer import hydra_trainer

logger = logging.getLogger(__name__)


class OptunaObjective:
    
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