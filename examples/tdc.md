# Therapeutic Data Commons (TDC) ADMET Benchmark

> \[!Note\]
> Check out the following runs' output presented as a [W&B report](https://api.wandb.ai/links/damin/f6iwfonl).

## Solubility AqSolDB

```bash
python ./litgnn/train.py \
    model=tdc/solubility_aqsoldb-cmpnn \
    dataset=tdc/solubility_aqsoldb \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++train.batch_size=16 \
    ++dataset.pre_transform.atom_messages=False
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
    dataset_name: solubility_aqsoldb
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
            job_type: tdc-solubility_aqsoldb
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
        steps_per_epoch: 316
```
</details>

## AMES

```bash
python ./litgnn/train.py \
    model=tdc/ames-cmpnn \
    dataset=tdc/ames \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++train.batch_size=16 \
    ++dataset.pre_transform.atom_messages=False
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
    dataset_name: ames
    num_classes: 1

model:
    _target_: litgnn.nn.models.graph.GraphLevelGNN
    in_channels: 133
    out_channels: 1
    edge_dim: 147
    num_ffn_layers: 2
    pooling_func_name: global_mean_pool
    model_cls: CMPNN
    hidden_channels: 64
    num_conv_layers: 2
    communicator: mlp
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
            job_type: tdc-ames
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
        steps_per_epoch: 291
```
</details>

## BBBP

```bash
python ./litgnn/train.py \
    model=tdc/bbb_martins-cmpnn \
    dataset=tdc/bbb_martins \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++train.batch_size=32 \
    ++dataset.pre_transform.atom_messages=False
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
    num_conv_layers: 4
    communicator: gru
    dropout: 0.0

train:
    num_seed_runs: 5
    seed: 1
    batch_size: 32
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

## Lipophilicity

```bash
python ./litgnn/train.py \
    model=tdc/lipophilicity_astrazeneca-cmpnn \
    dataset=tdc/lipophilicity_astrazeneca \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++train.batch_size=32 \
    ++dataset.pre_transform.atom_messages=False
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
    dataset_name: lipophilicity_astrazeneca
    num_classes: 1

model:
    _target_: litgnn.nn.models.graph.GraphLevelGNN
    in_channels: 133
    out_channels: 1
    edge_dim: 147
    num_ffn_layers: 2
    pooling_func_name: global_mean_pool
    model_cls: CMPNN
    hidden_channels: 300
    num_conv_layers: 3
    communicator: mlp
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
            job_type: tdc-lipophilicity_astrazeneca
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
        steps_per_epoch: 84
```
</details>

## LD50

```bash
python ./litgnn/train.py \
    model=tdc/ld50_zhu-cmpnn \
    dataset=tdc/ld50_zhu \
    ++train.dataset.split_sizes=[0.8,0.2,0.0] \
    ++train.num_seed_runs=5 \
    ++train.batch_size=32 \
    ++dataset.pre_transform.atom_messages=False
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
    hidden_channels: 512
    num_conv_layers: 3
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
        logger:
            _target_: pytorch_lightning.loggers.WandbLogger
            project: LitGNN
            group: CMPNN
            job_type: tdc-ld50_zhu
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
        steps_per_epoch: 125
```
</details>
