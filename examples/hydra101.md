# Hydra 101

## Understanding CLI Syntax

> \[!Note\]
> Before going through the examples below, read the [Override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) section from the Hydra docs.

### Overriding config values
```bash
python ./litgnn/train.py \
    model=cmpnn \
    dataset=tdc/bbb_martins \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++model.communicator=gru \
    ++dataset.pre_transform.atom_messages=True
```
where,
|Key|Value|Explanation|
|:-:|:-:|:-:|
|model|`cmpnn`|Model config present in [conf/model/cmpnn.yaml](../litgnn/conf/model/cmpnn.yaml) will be loaded.|
|dataset|`tdc/bbb_martins`|Dataset config present in [conf/dataset/tdc/bbb_martins.yaml](../litgnn/conf/dataset/tdc/bbb_martins.yaml) will be loaded.|
|++train.dataset.split_sizes|`[0.8,0.2,0.0]`|Override (++ sign) existing value of the `split_sizes`. TDC provides `train_val` and `test` sets. The `split_sizes` will be used to split `train_val` into `train` and `val`.|
|++train.num_seed_runs|`5`|Override (++ sign) existing value of the `num_seed_runs`.|
|++model.communicator|`gru`|Override (++ sign) existing value of the `communicator` present in the [cmpnn.yaml](../litgnn/conf/model/cmpnn.yaml) file.|
|++dataset.pre_transform.atom_messages|`True`|Override (++ sign) existing value of the `atom_messages` present in the [config.yaml](../litgnn/conf/config.yaml) file.|

<details>
<summary>View the hydra config that is passed to the trainer</summary>

```yaml
dataset:
    save_dir: .cache
    pre_transform:
        _target_: litgnn.nn.models.cmpnn.featurization.FeaturesGenerator
        atom_messages: true
    task:
        task_type: binary_classification
        loss:
        _target_: torch.nn.BCEWithLogitsLoss
        metrics:
        auroc:
            _target_: torchmetrics.AUROC
            task: binary
        auprc:
            _target_: torchmetrics.AveragePrecision
            task: binary
        f1score:
            _target_: torchmetrics.F1Score
            task: binary
    dataset_type: custom
    group_key: tdc
    dataset_name: bbb_martins
    num_classes: 1

model:
    _target_: litgnn.nn.models.graph.GraphLevelGNN
    in_channels: 133
    out_channels: 1
    edge_dim: 147
    num_ffn_layers: 2
    pooling_func_name: global_mean_pool
    model_cls: CMPNN
    hidden_channels: 256
    num_conv_layers: 3
    communicator: gru
    dropout: 0.0

train:
    num_seed_runs: 5
    seed: 1
    batch_size: 16
    dataset:
        split: scaffold_split
        split_sizes:
            - 0.8
            - 0.2
            - 0.0
        num_node_features: 133
        num_edge_features: 147
    trainer:
        _target_: pytorch_lightning.Trainer
        accelerator: gpu
        devices: auto
        max_epochs: 100
        log_every_n_steps: 10
        logger:
            _target_: pytorch_lightning.loggers.WandbLogger
            project: LitGNN
            group: CMPNN
            job_type: tdc-bbb_martins
            name: Seed_1
            dir: wandb
        callbacks:
            - _target_: pytorch_lightning.callbacks.RichProgressBar
            - _target_: pytorch_lightning.callbacks.ModelCheckpoint
              monitor: val_loss
              mode: min
              save_top_k: 1
              filename: '{epoch:02d}-{val_loss:.4f}'
            - _target_: pytorch_lightning.callbacks.EarlyStopping
              monitor: val_loss
              mode: min
              patience: 10
    optimizer:
        _target_: torch.optim.Adam
        lr: 0.0001
        weight_decay: 0
    scheduler:
        _target_: litgnn.nn.lr_scheduler.NoamLRScheduler
        warmup_epochs:
            - 2
        total_epochs:
            - 100
        init_lr:
            - 0.0001
        max_lr:
            - 0.001
        final_lr:
            - 0.0001
        steps_per_epoch: 41
```
</details>

### Appending to the defaults list

Add hyperparameter optimization default config and model specific hyperparameters config to the main config,
```bash
python ./litgnn/optimize.py \
    model=cmpnn \
    dataset=tdc/cyp2d6_veith \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    +hpo/optuna=default \
    hpo/optuna/search_spaces@hydra.sweeper.params=cmpnn \
    ++hydra.sweeper.n_trials=50 \
    ++optuna_lightning_pruning_callback.monitor=val_auprc \
    ++hydra.sweeper.direction=maximize
```
where,
|Key|Value|Explanation|
|:-:|:-:|:-:|
|model|`cmpnn`|Model config present in [conf/model/cmpnn.yaml](../litgnn/conf/model/cmpnn.yaml) will be loaded.|
|dataset|`tdc/cyp2d6_veith`|Dataset config present in [conf/dataset/tdc/cyp2d6_veith.yaml](../litgnn/conf/dataset/tdc/cyp2d6_veith.yaml) will be loaded.|
|++train.dataset.split_sizes|`[0.8,0.2,0.0]`|Override (++ sign) existing value of the `split_sizes`. TDC provides `train_val` and `test` sets. The `split_sizes` will be used to split `train_val` into `train` and `val`.|
|+hpo/optuna|`default`|Append (+ sign) [default](../litgnn/conf/hpo/optuna/default.yaml) optuna config.|
|hpo/optuna/search_spaces@hydra.sweeper.params|`cmpnn`|Add the model specific hparams to the default optuna config.|
|++hydra.sweeper.n_trials|`50`|Override (++ sign) exisiting value of `n_trials`.|
|++optuna_lightning_pruning_callback.monitor|`val_auprc`|Override (++ sign) exisiting value of `monitor`.|
|++hydra.sweeper.direction|`maximize`|Override (++ sign) exisiting value of `direction`.|

<details>
<summary>View the hydra config that is passed to the trainer</summary>

```yaml
hydra:
    run:
        dir: outputs/2024-04-20/11-06-46
    sweep:
        dir: multirun/2024-04-20/11-06-46
        subdir: ???
    launcher:
        _target_: hydra._internal.core_plugins.basic_launcher.BasicLauncher
    sweeper:
        sampler:
            _target_: optuna.samplers.TPESampler
            seed: 42
            consider_prior: true
            prior_weight: 1.0
            consider_magic_clip: true
            consider_endpoints: false
            n_startup_trials: 10
            n_ei_candidates: 24
            multivariate: false
            warn_independent_sampling: true
        _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
        direction: maximize
        storage: null
        study_name: CMPNN
        n_trials: 50
        n_jobs: 1
        search_space: null
        params:
            model.communicator: choice("additive", "inner_product", "gru", "mlp")
            model.num_conv_layers: range(2, 6, step=1)
            model.hidden_channels: choice(64, 128, 256, 512)
            train.batch_size: choice(16, 32, 64, 128, 256)
        custom_search_space: null
    #....other hydra config.....#

optuna_lightning_pruning_callback:
    monitor: val_auprc

dataset:
    save_dir: .cache
    pre_transform:
        _target_: litgnn.nn.models.cmpnn.featurization.FeaturesGenerator
        atom_messages: false
    task:
        task_type: binary_classification
        loss:
        _target_: torch.nn.BCEWithLogitsLoss
        metrics:
        auroc:
            _target_: torchmetrics.AUROC
            task: binary
        auprc:
            _target_: torchmetrics.AveragePrecision
            task: binary
        f1score:
            _target_: torchmetrics.F1Score
            task: binary
    dataset_type: custom
    group_key: tdc
    dataset_name: cyp2d6_veith
    num_classes: 1

model:
    _target_: litgnn.nn.models.graph.GraphLevelGNN
    in_channels: 133
    out_channels: 1
    edge_dim: 147
    num_ffn_layers: 2
    pooling_func_name: global_mean_pool
    model_cls: CMPNN
    hidden_channels: 512
    num_conv_layers: 4
    communicator: additive
    dropout: 0.0

train:
    num_seed_runs: 1
    seed: 1
    batch_size: 16
    dataset:
        split: scaffold_split
        split_sizes:
            - 0.8
            - 0.2
            - 0.0
        num_node_features: 133
        num_edge_features: 147
    trainer:
        _target_: pytorch_lightning.Trainer
        accelerator: gpu
        devices: auto
        max_epochs: 100
        log_every_n_steps: 10
        logger:
            _target_: pytorch_lightning.loggers.WandbLogger
            project: LitGNN
            group: CMPNN-Optuna
            job_type: tdc-cyp2d6_veith
            name: null
            dir: wandb
        callbacks:
            - _target_: pytorch_lightning.callbacks.RichProgressBar
            - _target_: pytorch_lightning.callbacks.ModelCheckpoint
              monitor: val_loss
              mode: min
              save_top_k: 1
              filename: '{epoch:02d}-{val_loss:.4f}'
            - _target_: pytorch_lightning.callbacks.EarlyStopping
              monitor: val_loss
              mode: min
              patience: 10
    optimizer:
        _target_: torch.optim.Adam
        lr: 0.0001
        weight_decay: 0
    scheduler:
        _target_: litgnn.nn.lr_scheduler.NoamLRScheduler
        warmup_epochs:
        - 2
        total_epochs:
        - 100
        init_lr:
        - 0.0001
        max_lr:
        - 0.001
        final_lr:
        - 0.0001
        steps_per_epoch: 526
```
</details>

### Removing a config value

Remove the default Weights and Biases (W&B) lightning logger,
```bash
python ./litgnn/train.py \
    model=tdc/ld50_zhu-cmpnn \
    dataset=tdc/ld50_zhu \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++train.batch_size=32 \
    ++dataset.pre_transform.atom_messages=False \
    ~train.trainer.logger
```

<details>
<summary>View the hydra config that is passed to the trainer</summary>

```yaml
dataset:
    save_dir: .cache
    pre_transform:
        _target_: litgnn.nn.models.cmpnn.featurization.FeaturesGenerator
        atom_messages: false
    task:
        task_type: regression
        loss:
            _target_: torch.nn.MSELoss
        metrics:
            mae:
                _target_: torchmetrics.MeanAbsoluteError
            r2score:
                _target_: torchmetrics.R2Score
    dataset_type: custom
    group_key: tdc
    dataset_name: ld50_zhu
    num_classes: 1

model:
    _target_: litgnn.nn.models.graph.GraphLevelGNN
    in_channels: 133
    out_channels: 1
    edge_dim: 147
    num_ffn_layers: 2
    pooling_func_name: global_mean_pool
    model_cls: CMPNN
    hidden_channels: 256
    num_conv_layers: 4
    communicator: additive
    dropout: 0.0

train:
    num_seed_runs: 5
    seed: 1
    batch_size: 16
    dataset:
        split: scaffold_split
        split_sizes:
            - 0.8
            - 0.2
            - 0.0
        num_node_features: 133
        num_edge_features: 147
    trainer:
        _target_: pytorch_lightning.Trainer
        accelerator: gpu
        devices: auto
        max_epochs: 100
        log_every_n_steps: 10
        callbacks:
            - _target_: pytorch_lightning.callbacks.RichProgressBar
            - _target_: pytorch_lightning.callbacks.ModelCheckpoint
              monitor: val_loss
              mode: min
              save_top_k: 1
              filename: '{epoch:02d}-{val_loss:.4f}'
            - _target_: pytorch_lightning.callbacks.EarlyStopping
              monitor: val_loss
              mode: min
              patience: 10
    optimizer:
        _target_: torch.optim.Adam
        lr: 0.0001
        weight_decay: 0
    scheduler:
        _target_: litgnn.nn.lr_scheduler.NoamLRScheduler
        warmup_epochs:
            - 2
        total_epochs:
            - 100
        init_lr:
            - 0.0001
        max_lr:
            - 0.001
        final_lr:
            - 0.0001
        steps_per_epoch: 125
```
</details>
