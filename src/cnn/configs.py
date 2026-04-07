from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Literal, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.cnn.cnn_paths import DATASET_DIR


@dataclass
class ModelConfig:
    name: str = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLoaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    train_shuffle: bool = True
    eval_shuffle: bool = False
    drop_last_train: bool = False
    drop_last_eval: bool = False


@dataclass
class LossConfig:
    cls: type[nn.Module] = nn.CrossEntropyLoss
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    cls: type[torch.optim.Optimizer] = torch.optim.Adam
    kwargs: dict[str, Any] = field(default_factory=lambda: {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    })


@dataclass
class SchedulerConfig:
    """
    step_metric examples:
      - None         -> scheduler.step() every epoch
      - "val/loss"   -> scheduler.step(epoch_metrics["val/loss"])
      - "val/accuracy"
    """
    cls: Optional[type] = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    step_metric: Optional[str] = None


@dataclass
class TrainConfig:
    epochs: int = 50
    device: str = "cpu"
    non_blocking: bool = True

    # mixed precision
    use_amp: bool = False

    # gradient clipping
    grad_clip_norm: Optional[float] = None

    # best checkpoint selection
    best_metric: str = "val/accuracy"
    best_mode: Literal["min", "max"] = "max"

    # optional reproducibility
    seed: Optional[int] = None


@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = "MPW-CNN"
    entity: Optional[str] = "MSE_DeLearn_SPR26"
    mode: str = "online"  # "online", "offline", "disabled"
    run_name: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    tags: list[str] = field(default_factory=list)
    notes: Optional[str] = None

    # what to log
    log_epoch_metrics: bool = True
    log_every_n_epochs: int = 1

    # if empty -> log everything passed to logger
    metric_allowlist: set[str] = field(default_factory=set)

    # same idea for summary
    summary_allowlist: set[str] = field(default_factory=set)

    # model watching
    watch_model: bool = False
    watch_log: str = "all"
    watch_log_freq: int = 100

    log_confusion_matrix: bool = True
    confusion_matrix_split: str = "val"   # "val" or "test"
    confusion_matrix_title: str | None = None
    confusion_matrix_split_table: bool = False

@dataclass
class DatasetConfig:
    dataset_dir: Path = DATASET_DIR
    train_subdir: str = "train"
    val_subdir: str = "validate"
    test_subdir: Optional[str] = None

    image_size: int = 224

    train_transform: Optional[transforms.Compose] = None
    eval_transform: Optional[transforms.Compose] = None

    normalize_mean: Optional[Sequence[float]] = None
    normalize_std: Optional[Sequence[float]] = None

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)




