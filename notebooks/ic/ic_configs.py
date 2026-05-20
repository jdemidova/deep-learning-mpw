# stdlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

# third-party
import torch.optim
from torch import nn
from torchvision import transforms


@dataclass
class LossConfig:
    cls: type[nn.Module] = nn.CrossEntropyLoss
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    cls: type[torch.optim.Optimizer] = torch.optim.AdamW
    kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
    )


@dataclass
class SchedulerConfig:
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
    grad_clip_norm: Optional[float] = 1.0

    # best checkpoint selection
    best_metric: str = "val/loss"
    best_mode: Literal["min", "max"] = "min"

    # early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # optional reproducibility
    seed: Optional[int] = 13

@dataclass
class GenerationConfig:
    max_len: int = 40
    decoding: Literal["greedy", "beam"] = "greedy"

    # Extension: beam search
    beam_size: int = 3
    length_penalty: float = 0.7

@dataclass
class EvaluationConfig:
    fixed_subset_size: int = 8
    compute_bleu_every_n_epochs: int = 1
    max_bleu_batches: Optional[int] = None

@dataclass
class WandBConfig:
    enabled: bool = True
    project: str = "MPW-IC"
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

    log_generated_examples: bool = True
    log_attention_examples: bool = False


@dataclass
class TokenizerConfig:
    tokenizer: Any = None
    min_word_freq: int = 5
    max_len: int = 40


@dataclass
class DatasetConfig:
    dataset_dir: Path = Path("/data")
    image_size: int = 224

    train_transform: Optional[transforms.Compose] = None
    eval_transform: Optional[transforms.Compose] = None

    normalize_mean: Optional[Sequence[float]] = None
    normalize_std: Optional[Sequence[float]] = None


@dataclass
class CaptionDataConfig:

    train_split: str = "train"
    test_split: str = "test"

    train_caption_sampling: Literal["all", "first", "random"] = "all"
    eval_caption_sampling: Literal["all", "first", "random"] = "all"
    generation_caption_sampling: Literal["all", "first", "random"] = "first"

@dataclass
class CaptionModelConfig:
    name: Literal["show_and_tell", "show_attend_and_tell"] = "show_and_tell"

    encoder_name: str = "resnet18"
    pretrained: bool = True
    freeze_encoder: bool = True

    embed_dim: int = 256
    hidden_dim: int = 512
    num_lstm_layers: int = 1
    dropout: float = 0.5

    # attention model only!
    attention_dim: int = 256

    # filled after tokenizer is built
    vocab_size: Optional[int] = None
    pad_idx: Optional[int] = None
    bos_idx: Optional[int] = None
    eos_idx: Optional[int] = None
    unk_idx: Optional[int] = None


@dataclass
class ICExperimentConfig:
    caption_data: CaptionDataConfig = field(default_factory=CaptionDataConfig)

    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    model: CaptionModelConfig = field(default_factory=CaptionModelConfig)

    loss: LossConfig = field(default_factory=LossConfig)

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    train: TrainConfig = field(default_factory=TrainConfig)

    generation: GenerationConfig = field(default_factory=GenerationConfig)

    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    wandb: WandBConfig = field(default_factory=WandBConfig)

def attach_tokenizer_to_model_config(model_cfg, tokenizer):
    """Copy tokenizer-dependent values into model config after tokenizer creation."""
    model_cfg.vocab_size = len(tokenizer)
    model_cfg.pad_idx = tokenizer.stoi["<pad>"]
    model_cfg.bos_idx = tokenizer.stoi["<bos>"]
    model_cfg.eos_idx = tokenizer.stoi["<eos>"]
    model_cfg.unk_idx = tokenizer.stoi["<unk>"]
    return model_cfg

def make_show_and_tell_config() -> ICExperimentConfig:
    cfg = ICExperimentConfig()

    cfg.model.name = "show_and_tell"
    cfg.wandb.group = "show_and_tell"
    cfg.wandb.tags = ["baseline", "resnet18", "lstm", "greedy"]

    return cfg


def make_show_attend_and_tell_config() -> ICExperimentConfig:
    cfg = ICExperimentConfig()

    cfg.model.name = "show_attend_and_tell"
    cfg.wandb.group = "show_attend_and_tell"
    cfg.wandb.tags = ["attention", "resnet18", "lstm", "greedy"]

    return cfg