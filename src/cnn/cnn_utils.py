from __future__ import annotations
import math
import copy
from datetime import datetime
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import wandb

from collections import Counter
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.cnn.configs import DatasetConfig
from src.cnn.cnn_registry import build_model
from src.cnn.configs import *
import src.cnn.cnn_paths as paths


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


def _effective_pin_memory(cfg: ExperimentConfig) -> bool:
    device_type = torch.device(cfg.train.device).type
    return bool(cfg.loader.pin_memory and device_type == "cuda")

def make_train_loader(dataset, cfg: ExperimentConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.train_shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=_effective_pin_memory(cfg),
        drop_last=cfg.loader.drop_last_train,
    )


def make_eval_loader(dataset, cfg: ExperimentConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.eval_shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=_effective_pin_memory(cfg),
        drop_last=cfg.loader.drop_last_eval,
    )

def configure_runtime_defaults(cfg: ExperimentConfig) -> ExperimentConfig:
    device = torch.device(cfg.train.device)

    # pin_memory is only meaningful here for CUDA in this codebase
    if device.type != "cuda":
        cfg.loader.pin_memory = False

    # AMP is only enabled here for CUDA
    if device.type != "cuda":
        cfg.train.use_amp = False

    return cfg

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
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    device = torch.device(cfg.train.device)
    use_amp = cfg.train.use_amp and device.type == "cuda"

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    start_time = time.perf_counter()

    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device, non_blocking=cfg.train.non_blocking)
        y = y_cpu.to(device, non_blocking=cfg.train.non_blocking).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        if use_amp:
            assert scaler is not None
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

        batch_size = y_cpu.size(0)
        running_loss += loss.item() * batch_size

        preds_cpu = logits.detach().argmax(dim=1).cpu()
        y_cpu = y_cpu.long().cpu()
        running_correct += (preds_cpu == y_cpu).sum().item()
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
def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    cfg: ExperimentConfig,
    prefix: str = "val",
) -> dict[str, float]:
    device = torch.device(cfg.train.device)
    criterion_cpu = build_loss(cfg, torch.device("cpu"))

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device, non_blocking=cfg.train.non_blocking)
        y = y_cpu.to(device, non_blocking=cfg.train.non_blocking).long()

        logits = model(x)

        logits_cpu = logits.detach().float().cpu()
        y_cpu = y_cpu.long().cpu()

        loss_cpu = criterion_cpu(logits_cpu, y_cpu)

        batch_size = y_cpu.size(0)
        running_loss += loss_cpu.item() * batch_size
        running_correct += (logits_cpu.argmax(dim=1) == y_cpu).sum().item()
        running_total += batch_size

    return {
        f"{prefix}/loss": running_loss / running_total,
        f"{prefix}/accuracy": running_correct / running_total,
    }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    cfg: ExperimentConfig,
    val_loader: Optional[DataLoader] = None,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device)
    model = model.to(device)

    criterion = build_loss(cfg, device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    use_amp = cfg.train.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
            scaler=scaler,
        )

        epoch_metrics: dict[str, float] = {"epoch": float(epoch)}
        epoch_metrics.update(train_metrics)

        if val_loader is not None:
            val_metrics = evaluate_model_on_loader(
                model=model,
                loader=val_loader,
                cfg=cfg,
                prefix="val",
            )
            epoch_metrics.update(val_metrics)
            epoch_metrics["gap/accuracy"] = epoch_metrics["train/accuracy"] - epoch_metrics["val/accuracy"]
            epoch_metrics["gap/loss"] = epoch_metrics["val/loss"] - epoch_metrics["train/loss"]

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

        epoch_metrics["lr"] = optimizer.param_groups[0]["lr"]

        if cfg.train.best_metric not in epoch_metrics:
            raise KeyError(
                f"Best metric '{cfg.train.best_metric}' not found in epoch metrics: "
                f"{list(epoch_metrics.keys())}"
            )

        current_value = epoch_metrics[cfg.train.best_metric]
        print(
            "epoch", epoch,
            "train_acc", epoch_metrics["train/accuracy"],
            "val_acc", epoch_metrics.get("val/accuracy"),
            "best_before", best_value,
            "current", current_value,
        )
        if _is_better(current_value, best_value, cfg.train.best_mode):
            best_value = current_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

            epoch_metrics["best/epoch_so_far"] = float(best_epoch)
            epoch_metrics[f"best/{cfg.train.best_metric}_so_far"] = float(best_value)
            print(">>> updating best_state at epoch", epoch, "to", current_value)

        _append_to_history(history, epoch_metrics)

        if cfg.wandb.log_epoch_metrics and (epoch % cfg.wandb.log_every_n_epochs == 0):
            log_metrics_to_wandb(epoch_metrics, cfg)

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

    model.load_state_dict(best_state)

    result = {
        "best_epoch": best_epoch,
        "best_metric_name": cfg.train.best_metric,
        "best_metric_value": best_value,
    }

    return model, history, result


def train_and_evaluate_model(
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

    run_id = run.id if run is not None else None

    trained_model, history, train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
    )
    trained_model = trained_model.to(torch.device(cfg.train.device))

    device = torch.device(cfg.train.device)
    criterion = build_loss(cfg, device)

    final_metrics: dict[str, float] = {}
    final_metrics.update(evaluate_model_on_loader(trained_model, train_loader, cfg, prefix="train_final"))

    if val_loader is not None:
        final_metrics.update(evaluate_model_on_loader(trained_model, val_loader, cfg, prefix="val_final"))

    if test_loader is not None:
        final_metrics.update(evaluate_model_on_loader(trained_model, test_loader, cfg, prefix="test"))

    result = {
        **train_result,
        **final_metrics,
    }

    result["wandb_run_id"] = run_id
    result["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # optional final one-shot log
    if cfg.wandb.enabled and cfg.wandb.mode != "disabled":
        log_metrics_to_wandb(final_metrics, cfg)
        write_summary_to_wandb(result, cfg)

    if torch.backends.mps.is_available() and str(cfg.train.device).startswith("mps"):
        torch.mps.synchronize()
        torch.mps.empty_cache()

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

    model, history, result = train_and_evaluate_model(
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

@torch.no_grad()
def predict_loader(model, loader, cfg, return_probs: bool = False):
    device = torch.device(cfg.train.device)

    model = model.to(device)

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for x, y in loader:
        x = x.to(device, non_blocking=cfg.train.non_blocking)
        y = y.to(device, non_blocking=cfg.train.non_blocking)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.extend(y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

        if return_probs:
            y_prob.extend(probs.cpu().tolist())

    if return_probs:
        return y_true, y_pred, y_prob
    return y_true, y_pred


def log_confusion_matrix(
    y_true,
    y_pred=None,
    y_prob=None,
    class_names=None,
    key: str = "val/confusion_matrix",
    title: str = "Validation Confusion Matrix",
    split_table: bool = False,
):
    if y_pred is None and y_prob is None:
        raise ValueError("Provide either y_pred or y_prob.")
    if y_pred is not None and y_prob is not None:
        raise ValueError("Provide only one of y_pred or y_prob.")

    y_true = [int(v) for v in y_true]

    if y_pred is not None:
        y_pred = [int(v) for v in y_pred]

    if class_names is not None:
        n_classes = len(class_names)

        bad_true = sorted(set(v for v in y_true if v < 0 or v >= n_classes))
        if bad_true:
            raise ValueError(
                f"y_true contains labels outside [0, {n_classes - 1}]: {bad_true}"
            )

        if y_pred is not None:
            bad_pred = sorted(set(v for v in y_pred if v < 0 or v >= n_classes))
            if bad_pred:
                raise ValueError(
                    f"y_pred contains labels outside [0, {n_classes - 1}]: {bad_pred}"
                )

        if y_prob is not None:
            prob_widths = sorted(set(len(row) for row in y_prob))
            if prob_widths != [n_classes]:
                raise ValueError(
                    f"y_prob width does not match class_names length: "
                    f"{prob_widths} vs {n_classes}"
                )

    if y_prob is not None:
        chart = wandb.plot.confusion_matrix(
            probs=y_prob,
            y_true=y_true,
            class_names=class_names,
            title=title,
            split_table=split_table,
        )
    else:
        chart = wandb.plot.confusion_matrix(
            preds=y_pred,
            y_true=y_true,
            class_names=class_names,
            title=title,
            split_table=split_table,
        )

    wandb.log({key: chart})


def make_confusion_matrix_fig(y_true, y_pred, class_names, normalize=None, title="Confusion Matrix"):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        normalize=normalize,
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _find_last_linear(model: nn.Module) -> tuple[str, nn.Linear]:
    base = _unwrap_model(model)

    last_name = None
    last_layer = None
    for name, layer in base.named_modules():
        if isinstance(layer, nn.Linear):
            last_name = name
            last_layer = layer

    if last_layer is None:
        raise ValueError("No nn.Linear layer found in model.")

    return last_name, last_layer

def manual_acc_debug(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == y.cpu()).sum().item()
            total += y.size(0)
    return correct, total, correct / total

def _unpack_sample(sample: Any) -> tuple[torch.Tensor, int]:
    """
    Supports:
      - (image, label)
      - (image, label, ...)
      - {"image": ..., "label": ...}
    """
    if isinstance(sample, dict):
        return sample["image"], int(sample["label"])

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], int(sample[1])

    raise TypeError(
        "Unsupported dataset sample format. Expected tuple/list with (image, label) "
        "or dict with keys 'image' and 'label'."
    )


def _default_id_getter(dataset, idx: int) -> str:
    """
    Best effort:
    - if sample is a dict and contains common id/path keys, use them
    - otherwise fallback to dataset index as string
    """
    sample = dataset[idx]

    if isinstance(sample, dict):
        for key in ("id", "image_id", "sample_id", "path", "filepath", "filename"):
            if key in sample:
                return str(sample[key])

    return str(idx)


def _identity_norm_params(num_channels: int) -> tuple[list[float], list[float]]:
    return [0.0] * num_channels, [1.0] * num_channels


def denorm_to_rgb(
    image_tensor: torch.Tensor,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> np.ndarray:
    """
    image_tensor: CHW tensor
    returns: HWC float image in [0, 1]
    """
    x = image_tensor.detach().cpu().float()

    if x.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape={tuple(x.shape)}")

    c = x.shape[0]
    if mean is None or std is None:
        mean, std = _identity_norm_params(c)

    mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)

    x = x * std_t + mean_t
    x = x.clamp(0, 1)

    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    return x.permute(1, 2, 0).numpy()


@torch.no_grad()
def collect_prediction_records(
    model: nn.Module,
    dataset,
    cfg,
    class_names: Sequence[str],
    *,
    batch_size: int = 64,
    id_getter: Callable[[Any, int], str] | None = None,
) -> pd.DataFrame:
    """
    Builds one row per sample with:
    - dataset index / sample id
    - true/pred labels
    - predicted confidence
    - true-class probability
    - confidence gap
    - entropy
    - correctness
    """
    device = torch.device(cfg.train.device)
    model.eval()

    if id_getter is None:
        id_getter = _default_id_getter

    rows: list[dict[str, Any]] = []

    for start in range(0, len(dataset), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(dataset))))

        xs = []
        ys = []

        for idx in batch_indices:
            x, y = _unpack_sample(dataset[idx])
            xs.append(x)
            ys.append(y)

        x_batch = torch.stack(xs).to(device, non_blocking=cfg.train.non_blocking)
        y_batch = torch.tensor(ys, dtype=torch.long)

        logits = model(x_batch)
        probs = torch.softmax(logits, dim=1).detach().cpu()

        pred_idx = probs.argmax(dim=1)
        pred_conf = probs.max(dim=1).values
        true_conf = probs[torch.arange(len(batch_indices)), y_batch]
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)

        # margin between top-1 and top-2 probs
        top2 = torch.topk(probs, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]

        for j, ds_idx in enumerate(batch_indices):
            y_true = int(y_batch[j].item())
            y_pred = int(pred_idx[j].item())

            rows.append(
                {
                    "index": ds_idx,
                    "sample_id": id_getter(dataset, ds_idx),
                    "true_idx": y_true,
                    "true_name": class_names[y_true],
                    "pred_idx": y_pred,
                    "pred_name": class_names[y_pred],
                    "correct": y_true == y_pred,
                    "pred_conf": float(pred_conf[j].item()),
                    "true_conf": float(true_conf[j].item()),
                    "conf_gap": float(pred_conf[j].item() - true_conf[j].item()),
                    "wrongness_score": float(pred_conf[j].item() - true_conf[j].item()),
                    "entropy": float(entropy[j].item()),
                    "margin": float(margin[j].item()),

                }
            )

    df = pd.DataFrame(rows)

    df["correct"] = df["correct"].astype(bool)
    return df

def top_n_most_confident_mistakes(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Samples where the model was wrong and highly certain.
    These are the most dangerous failures and often reveal shortcut features or systematic confusion."""
    n = min(n, 30)
    return (
        df.loc[~df["correct"]]
        .sort_values(["pred_conf", "entropy"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )


def top_n_least_confident_correct(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Samples the model got right but only barely. These are fragile wins near the decision boundary."""
    n = min(n, 30)
    return (
        df.loc[df["correct"]]
        .sort_values(["pred_conf", "entropy"], ascending=[True, False])
        .head(n)
        .reset_index(drop=True)
    )


def top_n_most_uncertain_overall(
    df: pd.DataFrame,
    n: int = 30,
    *,
    by: str = "pred_conf",
) -> pd.DataFrame:
    """
    Samples with the weakest class preference.
    These often indicate ambiguous images, overlapping classes, or a messy decision boundary.

    by='pred_conf'  -> lowest max probability first
    by='entropy'    -> highest entropy first
    by='margin'     -> smallest top1-top2 margin first
    """
    n = min(n, 30)

    if by == "pred_conf":
        return (
            df.sort_values(["pred_conf", "entropy"], ascending=[True, False])
            .head(n)
            .reset_index(drop=True)
        )
    if by == "entropy":
        return (
            df.sort_values(["entropy", "pred_conf"], ascending=[False, True])
            .head(n)
            .reset_index(drop=True)
        )
    if by == "margin":
        return (
            df.sort_values(["margin", "entropy"], ascending=[True, False])
            .head(n)
            .reset_index(drop=True)
        )

    raise ValueError("by must be one of: 'pred_conf', 'entropy', 'margin'")


def top_n_mistakes_per_true_class(
    df: pd.DataFrame,
    n: int = 30,
) -> dict[str, pd.DataFrame]:
    """The hardest failures for each true class. This avoids hiding class-specific errors behind aggregate metrics."""
    n = min(n, 30)
    out: dict[str, pd.DataFrame] = {}

    mistakes = df.loc[~df["correct"]].copy()

    for class_name, group in mistakes.groupby("true_name", sort=True):
        out[class_name] = (
            group.sort_values(["pred_conf", "entropy"], ascending=[False, True])
            .head(n)
            .reset_index(drop=True)
        )

    return out

def summarize_cases(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "index",
        "true_name",
        "pred_name",
        "correct",
        "pred_conf",
        "true_conf",
        "entropy",
        "margin",
    ]
    out = df[cols].copy()
    out["pred_conf"] = out["pred_conf"].round(4)
    out["true_conf"] = out["true_conf"].round(4)
    out["entropy"] = out["entropy"].round(4)
    out["margin"] = out["margin"].round(4)
    return out

def plot_case_grid(
    dataset,
    cases_df: pd.DataFrame,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    title: str | None = None,
    max_n: int = 30,
    ncols: int = 5,
    figsize_per_cell: tuple[float, float] = (3.4, 3.4),
) -> None:
    cases_df = cases_df.head(min(max_n, 30)).reset_index(drop=True)

    n = len(cases_df)
    if n == 0:
        print("No cases to plot.")
        return

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, cases_df.iterrows()):
        x, _ = _unpack_sample(dataset[int(row["index"])])
        img = denorm_to_rgb(x, mean=mean, std=std)

        ax.imshow(img)
        ax.set_title(
            "\n".join(
                [
                    f"id={row['sample_id']}",
                    f"true={row['true_name']}",
                    f"pred={row['pred_name']}",
                    f"p={row['pred_conf']:.3f}",
                ]
            ),
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        plt.tight_layout()

    plt.show()

def top_n_for_confusion_pair(
    df: pd.DataFrame,
    true_class: str | int,
    pred_class: str | int,
    *,
    n: int = 30,
    sort_by: str = "pred_conf",
) -> pd.DataFrame:
    """
    Filter rows for a specific confusion pair:
      true = true_class, pred = pred_class

    sort_by:
      - "pred_conf"       : most confident wrong predictions first
      - "wrongness_score" : biggest pred_conf - true_conf first
      - "entropy"         : most uncertain among that pair
      - "margin"          : smallest top1-top2 margin first
    """
    n = min(n, 30)

    if isinstance(true_class, str):
        mask_true = df["true_name"] == true_class
    else:
        mask_true = df["true_idx"] == true_class

    if isinstance(pred_class, str):
        mask_pred = df["pred_name"] == pred_class
    else:
        mask_pred = df["pred_idx"] == pred_class

    subset = df.loc[mask_true & mask_pred & (~df["correct"])].copy()

    if sort_by == "pred_conf":
        subset = subset.sort_values(["pred_conf", "entropy"], ascending=[False, True])
    elif sort_by == "wrongness_score":
        subset = subset.sort_values(["wrongness_score", "pred_conf"], ascending=[False, False])
    elif sort_by == "entropy":
        subset = subset.sort_values(["entropy", "pred_conf"], ascending=[False, False])
    elif sort_by == "margin":
        subset = subset.sort_values(["margin", "pred_conf"], ascending=[True, False])
    else:
        raise ValueError("sort_by must be one of: pred_conf, wrongness_score, entropy, margin")

    return subset.head(n).reset_index(drop=True)

def most_common_confusion_pairs(
    df: pd.DataFrame,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Returns the most frequent off-diagonal confusion pairs.
    """
    mistakes = df.loc[~df["correct"]].copy()

    if mistakes.empty:
        return pd.DataFrame(
            columns=["true_name", "pred_name", "count", "mean_pred_conf", "mean_wrongness_score"]
        )

    out = (
        mistakes.groupby(["true_name", "pred_name"], as_index=False)
        .agg(
            count=("index", "count"),
            mean_pred_conf=("pred_conf", "mean"),
            mean_wrongness_score=("wrongness_score", "mean"),
        )
        .sort_values(["count", "mean_pred_conf"], ascending=[False, False])
        .head(top_k)
        .reset_index(drop=True)
    )

    out["mean_pred_conf"] = out["mean_pred_conf"].round(4)
    out["mean_wrongness_score"] = out["mean_wrongness_score"].round(4)
    return out

def compare_prediction_tables(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    key: str = "sample_id",
) -> pd.DataFrame:
    """
    Compare per-sample predictions from two models/checkpoints.

    key:
      - "sample_id" preferred if stable
      - "index" okay if same dataset ordering
    """
    before = before_df.copy()
    after = after_df.copy()

    keep_cols = [
        key,
        "index",
        "true_idx",
        "true_name",
        "pred_idx",
        "pred_name",
        "correct",
        "pred_conf",
        "true_conf",
        "wrongness_score",
        "entropy",
        "margin",
    ]

    before = before[keep_cols].rename(
        columns={
            "pred_idx": "pred_idx_before",
            "pred_name": "pred_name_before",
            "correct": "correct_before",
            "pred_conf": "pred_conf_before",
            "true_conf": "true_conf_before",
            "wrongness_score": "wrongness_score_before",
            "entropy": "entropy_before",
            "margin": "margin_before",
        }
    )

    after = after[keep_cols].rename(
        columns={
            "pred_idx": "pred_idx_after",
            "pred_name": "pred_name_after",
            "correct": "correct_after",
            "pred_conf": "pred_conf_after",
            "true_conf": "true_conf_after",
            "wrongness_score": "wrongness_score_after",
            "entropy": "entropy_after",
            "margin": "margin_after",
        }
    )

    merged = before.merge(
        after,
        on=[key, "index", "true_idx", "true_name"],
        how="inner",
        validate="one_to_one",
    )

    merged["became_correct"] = (~merged["correct_before"]) & (merged["correct_after"])
    merged["became_wrong"] = (merged["correct_before"]) & (~merged["correct_after"])

    merged["delta_pred_conf"] = merged["pred_conf_after"] - merged["pred_conf_before"]
    merged["delta_true_conf"] = merged["true_conf_after"] - merged["true_conf_before"]
    merged["delta_wrongness_score"] = (
        merged["wrongness_score_after"] - merged["wrongness_score_before"]
    )
    merged["delta_entropy"] = merged["entropy_after"] - merged["entropy_before"]
    merged["delta_margin"] = merged["margin_after"] - merged["margin_before"]

    return merged

def top_n_fixed_cases(comp_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Samples that were wrong before and became correct after.
    Prioritize cases where true-class confidence improved the most.
    """
    n = min(n, 30)
    return (
        comp_df.loc[comp_df["became_correct"]]
        .sort_values(["delta_true_conf", "pred_conf_after"], ascending=[False, False])
        .head(n)
        .reset_index(drop=True)
    )


def top_n_regressions(comp_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Samples that were correct before and became wrong after.
    """
    n = min(n, 30)
    return (
        comp_df.loc[comp_df["became_wrong"]]
        .sort_values(["delta_wrongness_score", "pred_conf_after"], ascending=[False, False])
        .head(n)
        .reset_index(drop=True)
    )


def top_n_still_boldly_wrong_after(comp_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Cases wrong in both versions, but still very confidently wrong after.
    """
    n = min(n, 30)
    subset = comp_df.loc[(~comp_df["correct_before"]) & (~comp_df["correct_after"])].copy()
    return (
        subset.sort_values(["wrongness_score_after", "pred_conf_after"], ascending=[False, False])
        .head(n)
        .reset_index(drop=True)
    )


def summarize_comparison_cases(comp_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "sample_id",
        "index",
        "true_name",
        "pred_name_before",
        "pred_name_after",
        "correct_before",
        "correct_after",
        "pred_conf_before",
        "pred_conf_after",
        "true_conf_before",
        "true_conf_after",
        "wrongness_score_before",
        "wrongness_score_after",
    ]
    out = comp_df[cols].copy()

    for c in [
        "pred_conf_before",
        "pred_conf_after",
        "true_conf_before",
        "true_conf_after",
        "wrongness_score_before",
        "wrongness_score_after",
    ]:
        out[c] = out[c].round(4)

    return out


def comparison_rows_to_case_grid_df(
    comp_df: pd.DataFrame,
    *,
    use_after: bool = True,
) -> pd.DataFrame:
    """
    Convert comparison rows into the same shape expected by plot_case_grid().
    """
    if use_after:
        pred_name_col = "pred_name_after"
        pred_conf_col = "pred_conf_after"
    else:
        pred_name_col = "pred_name_before"
        pred_conf_col = "pred_conf_before"

    return pd.DataFrame(
        {
            "index": comp_df["index"],
            "sample_id": comp_df["sample_id"],
            "true_name": comp_df["true_name"],
            "pred_name": comp_df[pred_name_col],
            "pred_conf": comp_df[pred_conf_col],
        }
    )

