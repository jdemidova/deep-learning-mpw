from __future__ import annotations

import copy
import time
import random

from matplotlib import pyplot as plt

import src.cnn.cnn_paths as paths

import torch
import wandb

from src.cnn.cnn_registry import build_model
from src.cnn.configs import *

from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.cnn.configs import DatasetConfig


def _build_default_transform(
    image_size: int,
    normalize_mean=None,
    normalize_std=None,
):
    steps = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]

    if normalize_mean is not None and normalize_std is not None:
        steps.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

    return transforms.Compose(steps)


def load_datasets(cfg: DatasetConfig):
    """
    Build train/val/test ImageFolder datasets from config.

    Returns:
        dict[str, torchvision.datasets.ImageFolder]
        Keys present depend on which subdirs exist / are configured:
        'train', 'val', optionally 'test'
    """
    dataset_dir = Path(cfg.dataset_dir)

    train_transform = cfg.train_transform or _build_default_transform(
        image_size=cfg.image_size,
        normalize_mean=cfg.normalize_mean,
        normalize_std=cfg.normalize_std,
    )

    eval_transform = cfg.eval_transform or _build_default_transform(
        image_size=cfg.image_size,
        normalize_mean=cfg.normalize_mean,
        normalize_std=cfg.normalize_std,
    )

    datasets_dict = {}

    train_dir = dataset_dir / cfg.train_subdir
    val_dir = dataset_dir / cfg.val_subdir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory does not exist: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory does not exist: {val_dir}")

    datasets_dict["train"] = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform,
    )

    datasets_dict["val"] = datasets.ImageFolder(
        root=val_dir,
        transform=eval_transform,
    )

    if cfg.test_subdir is not None:
        test_dir = dataset_dir / cfg.test_subdir
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

        datasets_dict["test"] = datasets.ImageFolder(
            root=test_dir,
            transform=eval_transform,
        )

    return datasets_dict


def get_dataset_info(dataset) -> dict:
    """
    Return dataset metadata and class distribution for an ImageFolder dataset.

    Returns:
        {
            'num_samples': int,
            'num_classes': int,
            'classes': list[str],
            'class_to_idx': dict[str, int],
            'class_counts': dict[str, int],
        }
    """
    if not hasattr(dataset, "samples"):
        raise TypeError("get_dataset_info currently expects an ImageFolder-like dataset with .samples")

    class_counts_idx = Counter(label for _, label in dataset.samples)
    class_counts = {
        class_name: class_counts_idx[idx]
        for class_name, idx in dataset.class_to_idx.items()
    }

    return {
        "num_samples": len(dataset),
        "num_classes": len(dataset.classes),
        "classes": list(dataset.classes),
        "class_to_idx": dict(dataset.class_to_idx),
        "class_counts": class_counts,
    }


def print_dataset_info(dataset, to_plot=False, title: str = "Dataset") -> None:
    """
    Pretty-print dataset info returned from ImageFolder.
    """
    info = get_dataset_info(dataset)

    print(f"{title}:")
    print(f"  num_samples: {info['num_samples']}")
    print(f"  num_classes: {info['num_classes']}")
    print("  class distribution:")
    for cls_name, count in sorted(info["class_counts"].items()):
        print(f"    {cls_name}: {count}")

    class_names = dataset.classes

    # store first image per class
    seen = {}

    for i in range(len(dataset)):
        img, label = dataset[i]

        if label not in seen:
            seen[label] = img

        if len(seen) == len(class_names):
            break
    if to_plot:
        plt.figure(figsize=(12, 6))

        for idx, (label, img) in enumerate(seen.items()):
            plt.subplot(2, 5, idx + 1)

            # handle tensor vs PIL
            if hasattr(img, "permute"):
                img = img.permute(1, 2, 0)

            plt.imshow(img)
            plt.title(class_names[label])
            plt.axis("off")

        plt.tight_layout()
        plt.show()


def get_device(prefer: str = "auto") -> torch.device:
    """
    Select the best available device.

    Args:
        prefer:
            - 'auto': CUDA > MPS > CPU
            - 'cuda': require CUDA if available, else fallback
            - 'mps': require MPS if available, else fallback
            - 'cpu': always CPU
    """
    prefer = prefer.lower()

    if prefer == "cpu":
        return torch.device("cpu")

    if prefer in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_num_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in a model.

    Args:
        trainable_only:
            If True, count only parameters with requires_grad=True.
            If False, count all parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_train_loader(dataset, cfg: ExperimentConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.train_shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=cfg.loader.drop_last_train,
    )


def make_eval_loader(dataset, cfg: ExperimentConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.eval_shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=cfg.loader.drop_last_eval,
    )



def build_loss(cfg: ExperimentConfig, device: torch.device) -> nn.Module:
    return cfg.loss.cls(**cfg.loss.kwargs).to(device)


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    return cfg.optimizer.cls(model.parameters(), **cfg.optimizer.kwargs)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
):
    if cfg.scheduler.cls is None:
        return None
    return cfg.scheduler.cls(optimizer, **cfg.scheduler.kwargs)


def wandb_config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    model = build_model(cfg.model)
    num_params = get_num_parameters(model)
    return {
        "dataset": {
            "dataset_dir": str(cfg.dataset.dataset_dir),
            "train_subdir": cfg.dataset.train_subdir,
            "val_subdir": cfg.dataset.val_subdir,
            "test_subdir": cfg.dataset.test_subdir,
            "image_size": cfg.dataset.image_size,
            "normalize_mean": cfg.dataset.normalize_mean,
            "normalize_std": cfg.dataset.normalize_std,
            "train_transform_repr": str(cfg.dataset.train_transform),
            "eval_transform_repr": str(cfg.dataset.eval_transform),
        },
        "model": {
            "name": cfg.model.name,
            "kwargs": cfg.model.kwargs,
            "num_params": num_params,
        },
        "loader": {
            "batch_size": cfg.loader.batch_size,
            "num_workers": cfg.loader.num_workers,
            "pin_memory": cfg.loader.pin_memory,
            "train_shuffle": cfg.loader.train_shuffle,
            "eval_shuffle": cfg.loader.eval_shuffle,
            "drop_last_train": cfg.loader.drop_last_train,
            "drop_last_eval": cfg.loader.drop_last_eval,
        },
        "loss": {
            "cls": cfg.loss.cls.__name__,
            "kwargs": cfg.loss.kwargs,
        },
        "optimizer": {
            "cls": cfg.optimizer.cls.__name__,
            "kwargs": cfg.optimizer.kwargs,
        },
        "scheduler": {
            "cls": None if cfg.scheduler.cls is None else cfg.scheduler.cls.__name__,
            "kwargs": cfg.scheduler.kwargs,
            "step_metric": cfg.scheduler.step_metric,
        },
        "train": {
            "epochs": cfg.train.epochs,
            "device": cfg.train.device,
            "non_blocking": cfg.train.non_blocking,
            "use_amp": cfg.train.use_amp,
            "grad_clip_norm": cfg.train.grad_clip_norm,
            "best_metric": cfg.train.best_metric,
            "best_mode": cfg.train.best_mode,
            "seed": cfg.train.seed,
        },
        "wandb": {
            "enabled": cfg.wandb.enabled,
            "project": cfg.wandb.project,
            "entity": cfg.wandb.entity,
            "mode": cfg.wandb.mode,
            "run_name": cfg.wandb.run_name,
            "group": cfg.wandb.group,
            "job_type": cfg.wandb.job_type,
            "tags": cfg.wandb.tags,
            "notes": cfg.wandb.notes,
            "log_epoch_metrics": cfg.wandb.log_epoch_metrics,
            "log_every_n_epochs": cfg.wandb.log_every_n_epochs,
            "metric_allowlist": sorted(cfg.wandb.metric_allowlist),
            "summary_allowlist": sorted(cfg.wandb.summary_allowlist),
            "watch_model": cfg.wandb.watch_model,
            "watch_log": cfg.wandb.watch_log,
            "watch_log_freq": cfg.wandb.watch_log_freq,
        },
    }

def init_wandb_run(
    cfg: ExperimentConfig,
    model: Optional[nn.Module] = None,
    extra_config: Optional[dict[str, Any]] = None,
):
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return None

    run_cfg = wandb_config_to_dict(cfg)
    if extra_config:
        run_cfg.update(extra_config)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.wandb.run_name,
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        config=run_cfg,
        settings=wandb.Settings(init_timeout=300),
    )

    if model is not None and cfg.wandb.watch_model:
        wandb.watch(model, log=cfg.wandb.watch_log, log_freq=cfg.wandb.watch_log_freq)

    return run


def _filter_metrics_for_wandb(metrics: dict[str, float], cfg: ExperimentConfig) -> dict[str, float]:
    if not cfg.wandb.metric_allowlist:
        return metrics

    filtered = {}
    for k, v in metrics.items():
        if k == "epoch" or k in cfg.wandb.metric_allowlist:
            filtered[k] = v
    return filtered


def log_metrics_to_wandb(metrics: dict[str, float], cfg: ExperimentConfig) -> None:
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return
    if not cfg.wandb.log_epoch_metrics:
        return

    filtered = _filter_metrics_for_wandb(metrics, cfg)
    if filtered:
        wandb.log(filtered)


def write_summary_to_wandb(summary: dict[str, float], cfg: ExperimentConfig) -> None:
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return

    if cfg.wandb.summary_allowlist:
        summary = {k: v for k, v in summary.items() if k in cfg.wandb.summary_allowlist}

    for k, v in summary.items():
        wandb.summary[k] = v


def _append_to_history(history: dict[str, list[float]], metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        history.setdefault(k, []).append(v)


def _is_better(current: float, best: float, mode: Literal["min", "max"]) -> bool:
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"Unsupported best mode: {mode}")


# ============================================================================
# Core methods
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    cfg: ExperimentConfig,
) -> dict[str, float]:
    device = torch.device(cfg.train.device)
    use_amp = cfg.train.use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    start_time = time.perf_counter()

    for x, y in loader:
        x = x.to(device, non_blocking=cfg.train.non_blocking)
        y = y.to(device, non_blocking=cfg.train.non_blocking)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            scaler.scale(loss).backward()

            if cfg.train.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if cfg.train.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)

            optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    epoch_time = time.perf_counter() - start_time

    return {
        "train/loss": epoch_loss,
        "train/accuracy": epoch_acc,
        "train/time_sec": epoch_time,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    cfg: ExperimentConfig,
    prefix: str = "val",
) -> dict[str, float]:
    device = torch.device(cfg.train.device)

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=cfg.train.non_blocking)
        y = y.to(device, non_blocking=cfg.train.non_blocking)

        logits = model(x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == y).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total

    return {
        f"{prefix}/loss": epoch_loss,
        f"{prefix}/accuracy": epoch_acc,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: ExperimentConfig,
    val_loader: Optional[DataLoader] = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    """
    Training loop only.
    - builds loss / optimizer / scheduler from config
    - tracks best model according to cfg.train.best_metric
    - logs epoch metrics to W&B if enabled
    """
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device)
    model = model.to(device)

    criterion = build_loss(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    history: dict[str, list[float]] = {}

    best_value = float("-inf") if cfg.train.best_mode == "max" else float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            cfg=cfg,
        )

        epoch_metrics: dict[str, float] = {"epoch": float(epoch)}
        epoch_metrics.update(train_metrics)

        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                cfg=cfg,
                prefix="val",
            )
            epoch_metrics.update(val_metrics)

            # useful overfitting diagnostics
            epoch_metrics["gap/accuracy"] = epoch_metrics["train/accuracy"] - epoch_metrics["val/accuracy"]
            epoch_metrics["gap/loss"] = epoch_metrics["val/loss"] - epoch_metrics["train/loss"]

        # scheduler stepping
        if scheduler is not None:
            if cfg.scheduler.step_metric is None:
                scheduler.step()
            else:
                if cfg.scheduler.step_metric not in epoch_metrics:
                    raise KeyError(
                        f"Scheduler step metric '{cfg.scheduler.step_metric}' "
                        f"not found in epoch metrics: {list(epoch_metrics.keys())}"
                    )
                scheduler.step(epoch_metrics[cfg.scheduler.step_metric])

        # lr logging
        epoch_metrics["lr"] = optimizer.param_groups[0]["lr"]

        # best checkpoint selection
        if cfg.train.best_metric not in epoch_metrics:
            raise KeyError(
                f"Best metric '{cfg.train.best_metric}' not found in epoch metrics: "
                f"{list(epoch_metrics.keys())}"
            )

        current_value = epoch_metrics[cfg.train.best_metric]
        if _is_better(current_value, best_value, cfg.train.best_mode):
            best_value = current_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            epoch_metrics["best/epoch_so_far"] = float(best_epoch)
            epoch_metrics[f"best/{cfg.train.best_metric}_so_far"] = float(best_value)

        _append_to_history(history, epoch_metrics)

        if cfg.wandb.log_epoch_metrics and (epoch % cfg.wandb.log_every_n_epochs == 0):
            log_metrics_to_wandb(epoch_metrics, cfg)

        # console output
        msg = (
            f"Epoch [{epoch:03d}/{cfg.train.epochs:03d}] "
            f"| train_loss={epoch_metrics['train/loss']:.4f} "
            f"| train_acc={epoch_metrics['train/accuracy']:.4f} "
        )
        if val_loader is not None:
            msg += (
                f"| val_loss={epoch_metrics['val/loss']:.4f} "
                f"| val_acc={epoch_metrics['val/accuracy']:.4f} "
            )
        msg += f"| time={epoch_metrics['train/time_sec']:.1f}s"
        print(msg)

    # restore best weights
    model.load_state_dict(best_state)

    result = {
        "best_epoch": best_epoch,
        "best_metric_name": cfg.train.best_metric,
        "best_metric_value": best_value,
    }

    return model, history, result


def train_eval(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: ExperimentConfig,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    run_name: Optional[str] = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    """
    Full pipeline:
      1. init W&B run (optional)
      2. train
      3. evaluate best model on train / val / test
      4. write W&B summary (optional)
    """
    if run_name is not None:
        cfg.wandb.run_name = run_name

    run = init_wandb_run(cfg, model=model)

    trained_model, history, train_result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
    )

    device = torch.device(cfg.train.device)
    criterion = build_loss(cfg, device)

    final_metrics: dict[str, float] = {}
    final_metrics.update(evaluate(trained_model, train_loader, criterion, cfg, prefix="train_final"))

    if val_loader is not None:
        final_metrics.update(evaluate(trained_model, val_loader, criterion, cfg, prefix="val_final"))

    if test_loader is not None:
        final_metrics.update(evaluate(trained_model, test_loader, criterion, cfg, prefix="test"))

    result = {
        **train_result,
        **final_metrics,
    }

    # optional final one-shot log
    if cfg.wandb.enabled and cfg.wandb.mode != "disabled":
        log_metrics_to_wandb(final_metrics, cfg)
        write_summary_to_wandb(result, cfg)

    if run is not None:
        wandb.finish()

    return trained_model, history, result

def get_dataset(imagesize=224):
    """Load the training and validation datasets for image classification.\n
    Args:
        imagesize (int): The desired size to which all images will be resized. Default is 224.
        Can be set to 128 or 64 for faster training but may reduce accuracy.
    Returns:
        train_dataset (torchvision.datasets.ImageFolder): The training dataset.
        val_dataset (torchvision.datasets.ImageFolder): The validation dataset.
    """
    # Original image size 224x224, can be changed to 128x128 or 64x64 for faster training but may reduce accuracy
    size = imagesize

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root=paths.TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=paths.VAL_DIR,
        transform=val_transform
    )

    return train_dataset, val_dataset

def run_experiment(cfg: ExperimentConfig):
    datasets_dict = load_datasets(cfg.dataset)

    train_dataset = datasets_dict["train"]
    val_dataset = datasets_dict["val"]
    test_dataset = datasets_dict.get("test")

    print_dataset_info(train_dataset, title="Train dataset")
    print()
    print_dataset_info(val_dataset, title="Validation dataset")
    print()

    train_loader = make_train_loader(train_dataset, cfg)
    val_loader = make_eval_loader(val_dataset, cfg)
    test_loader = make_eval_loader(test_dataset, cfg) if test_dataset is not None else None

    model = build_model(cfg.model)

    print(model)
    print(f"Trainable parameters: {get_num_parameters(model):,}")

    model, history, result = train_eval(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        run_name=cfg.wandb.run_name,
    )

    return model, history, result

def save_checkpoint(path, filename, model, cfg, extra=None):

    payload = {
        "model_name": cfg.model.name,
        "run_name": cfg.wandb.run_name,
        "model_kwargs": cfg.model.kwargs,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path) / filename)

def log_checkpoint_artifact(path, filename, cfg, result=None, aliases=None):
    artifact = wandb.Artifact(
        name=f"{cfg.model.name}_checkpoint",
        type="model",
        metadata={
            "model_name": cfg.model.name,
            "run_name": cfg.wandb.run_name,
            "model_kwargs": cfg.model.kwargs,
        },
    )
    artifact.add_file(Path(path) / filename)
    if aliases is not None:
        artifact.add_file(aliases)
    wandb.log_artifact(artifact, aliases=aliases or ["latest"])

def load_model_for_inference(path, filename, map_location="cpu"):
    checkpoint = torch.load(Path(path) / filename, map_location=map_location, weights_only=True)

    cfg = ExperimentConfig()
    cfg.model.name = checkpoint["model_name"]
    cfg.model.kwargs = checkpoint["model_kwargs"]

    model = build_model(cfg.model)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint