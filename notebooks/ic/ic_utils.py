from __future__ import annotations

# stdlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# third-party
import numpy as np
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm.auto import tqdm

# local project imports
from notebooks.ic.ic_configs import ICExperimentConfig
from tokenizer import tokenize


# =========================================================
# GENERAL HELPERS
# =========================================================
def seed_everything(seed: int = 13) -> None:
    """Seed Python, NumPy, and PyTorch so repeated runs are more comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Mostly relevant for CUDA, MPS can still have some nondeterminism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg=None) -> torch.device:
    """Return the configured device, or choose the best available accelerator."""
    if cfg is not None:
        device_name = getattr(cfg.train, "device", "auto")
        if device_name not in (None, "auto"):
            return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def move_batch_to_device(
    batch,
    device: torch.device,
    non_blocking: bool = True,
):
    """
    Move tensor fields from a dataloader batch to the chosen device.

    The dataset batch also contains image IDs and raw captions. Those are Python
    metadata, not tensors, so they stay on the CPU unchanged.
    """
    images, captions, lengths, image_ids, raw_captions = batch

    return (
        images.to(device, non_blocking=non_blocking),
        captions.to(device, non_blocking=non_blocking),
        lengths.to(device, non_blocking=non_blocking),
        image_ids,
        raw_captions,
    )


def truncate_after_eos(token_ids: list[int], eos_idx: int) -> list[int]:
    """Drop generated tokens after the first <eos> token."""
    if eos_idx in token_ids:
        return token_ids[: token_ids.index(eos_idx) + 1]
    return token_ids


# =========================================================
# TRAIN HELPERS
# =========================================================

def split_caption_inputs_targets(captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shift captions for teacher-forced next-token prediction.

    caption:  <bos> a dog runs <eos>
    input:    <bos> a dog runs
    target:        a dog runs <eos>

    The model sees the prefix and is trained to predict the next token at each
    position. <eos> remains in the target so the model learns when to stop.
    """
    return captions[:, :-1].contiguous(), captions[:, 1:].contiguous()


def extract_logits(model_output: Any) -> torch.Tensor:
    """
    Return logits from common model output formats.

    This lets the shared training loop support simple models returning only
    logits and richer models returning extra values such as attention weights.
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output

def _run_teacher_forced_epoch(
    model: nn.Module,
    loader,
    cfg,
    desc: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """
    Run one teacher-forced epoch for either training or evaluation.

    Passing an optimizer enables gradients and parameter updates. Passing
    optimizer=None switches to evaluation mode and disables gradient tracking.
    """
    device = resolve_device(cfg)
    pad_idx = cfg.model.pad_idx
    non_blocking = cfg.train.non_blocking
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss_sum = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        # Use one loop for train and eval; only the train path builds gradients.
        with torch.set_grad_enabled(is_train):
            images, captions, lengths, image_ids, raw_captions = move_batch_to_device(
                batch,
                device=device,
                non_blocking=non_blocking,
            )

            decoder_inputs, targets = split_caption_inputs_targets(captions)

            if is_train:
                # set_to_none=True is a small memory/performance optimization.
                optimizer.zero_grad(set_to_none=True)

            model_output = model(images, decoder_inputs)
            loss, num_tokens = compute_captioning_loss(
                model_output,
                targets,
                pad_idx=pad_idx,
            )

            if is_train:
                loss.backward()

                if cfg.train.grad_clip_norm is not None:
                    # LSTMs can have unstable gradients; clipping limits spikes.
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=cfg.train.grad_clip_norm,
                    )

                optimizer.step()

        # loss is averaged over non-pad tokens, so weight it back by token count
        # before accumulating epoch-level metrics.
        total_loss_sum += float(loss.item()) * num_tokens
        total_tokens += num_tokens

        avg_loss = total_loss_sum / max(total_tokens, 1)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            ppl=f"{perplexity_from_loss(avg_loss):.2f}",
        )

    avg_loss = total_loss_sum / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity_from_loss(avg_loss),
    }

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg,
    desc: str = "train",
) -> dict[str, float]:
    """Train the model for one teacher-forced captioning epoch."""
    return _run_teacher_forced_epoch(
        model=model,
        loader=loader,
        cfg=cfg,
        desc=desc,
        optimizer=optimizer,
    )

# =========================================================
# EVAL HELPERS
# =========================================================

def decode_token_ids(
    token_ids: list[int] | torch.Tensor,
    tokenizer,
    eos_idx: int,
    skip_special_tokens: bool = True,
) -> str:
    """
    Convert generated token IDs into a readable caption string.

    Generation may return extra tokens after <eos>; those are ignored before
    passing IDs to the tokenizer.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()

    token_ids = [int(t) for t in token_ids]
    token_ids = truncate_after_eos(token_ids, eos_idx=eos_idx)

    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def token_ids_to_bleu_tokens(
    token_ids: list[int] | torch.Tensor,
    tokenizer,
    eos_idx: int,
) -> list[str]:
    """Convert generated token IDs into tokenizer words used by BLEU."""
    decoded = decode_token_ids(
        token_ids,
        tokenizer=tokenizer,
        eos_idx=eos_idx,
        skip_special_tokens=True,
    )
    return tokenize(decoded)

def perplexity_from_loss(loss: float) -> float:
    """
    Convert mean cross-entropy loss to perplexity.

    The loss is clipped before exp() to avoid huge numbers if training diverges.
    """
    return float(math.exp(min(loss, 20.0)))

def compute_captioning_loss(
    model_output: Any,
    targets: torch.Tensor,
    pad_idx: int,
) -> tuple[torch.Tensor, int]:
    """
    Compute masked next-token cross-entropy for captioning.

    Only <pad> targets are ignored. <eos> is intentionally kept because it is a
    real target token and teaches the decoder to terminate captions.
    """
    logits = extract_logits(model_output)
    vocab_size = logits.size(-1)

    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )

    # Average over real caption tokens, not over padded sequence length.
    num_tokens = int((targets != pad_idx).sum().item())
    loss = loss_sum / max(num_tokens, 1)

    return loss, num_tokens

def evaluate_loss(
    model: nn.Module,
    loader,
    cfg,
    desc: str = "eval",
) -> dict[str, float]:
    """
    Evaluate teacher-forced validation/test loss and perplexity.

    This uses the same shifted-input loss as training but does not update model
    weights.
    """
    return _run_teacher_forced_epoch(
        model=model,
        loader=loader,
        cfg=cfg,
        desc=desc,
        optimizer=None,
    )

@torch.no_grad()
def evaluate_bleu(
    model: nn.Module,
    image_loader,
    references_by_image_id: dict[str, list[list[str]]],
    tokenizer,
    cfg,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4 for generated captions.

    Generates one caption per image and compares it against all human references
    for the same image.
    """
    eos_idx = cfg.model.eos_idx
    hypotheses = []
    references = []

    for batch_idx, batch in enumerate(tqdm(image_loader, desc="BLEU generation", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, _, _, image_ids, _ = batch
        generated_ids = generate_batch_token_ids(model, images, cfg)

        for image_id, pred_ids in zip(image_ids, generated_ids):
            # BLEU expects every hypothesis to have a list of valid references.
            refs = references_by_image_id.get(image_id)

            if not refs:
                continue

            hyp = token_ids_to_bleu_tokens(
                pred_ids,
                tokenizer=tokenizer,
                eos_idx=eos_idx,
            )

            if len(hyp) == 0:
                # NLTK BLEU cannot score an empty hypothesis.
                hyp = ["<empty>"]

            references.append(refs)
            hypotheses.append(hyp)

    if not hypotheses:
        return {
            "bleu_1": 0.0,
            "bleu_2": 0.0,
            "bleu_3": 0.0,
            "bleu_4": 0.0,
        }

    # Smoothing avoids zero BLEU for short captions with missing higher n-grams.
    smoothing = SmoothingFunction().method1

    return {
        "bleu_1": corpus_bleu(
            references,
            hypotheses,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothing,
        ),
        "bleu_2": corpus_bleu(
            references,
            hypotheses,
            weights=(0.5, 0.5, 0.0, 0.0),
            smoothing_function=smoothing,
        ),
        "bleu_3": corpus_bleu(
            references,
            hypotheses,
            weights=(1 / 3, 1 / 3, 1 / 3, 0.0),
            smoothing_function=smoothing,
        ),
        "bleu_4": corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ),
    }

def collect_fixed_eval_subset(loader, cfg=None, num_images: int | None = None):
    """
    Collect fixed images for qualitative comparison.

    Same subset should be used for baseline, attention model, greedy decoding, beam search, etc.
    """
    if num_images is None:
        num_images = cfg.evaluation.fixed_subset_size if cfg is not None else 8

    collected_images = []
    collected_image_ids = []
    collected_raw_captions = []

    for batch in loader:
        images, _, _, image_ids, raw_captions = batch

        for image, image_id, raw_caption in zip(images, image_ids, raw_captions):
            collected_images.append(image)
            collected_image_ids.append(image_id)
            collected_raw_captions.append(raw_caption)

            if len(collected_images) >= num_images:
                return {
                    "images": torch.stack(collected_images),
                    "image_ids": collected_image_ids,
                    "raw_captions": collected_raw_captions,
                }

    return {
        "images": torch.stack(collected_images),
        "image_ids": collected_image_ids,
        "raw_captions": collected_raw_captions,
    }

def build_references_by_image_id(loader) -> dict[str, list[list[str]]]:
    """
    Build BLEU references from all human captions.

    Use this with a deterministic test loader using caption_sampling='all'.
    The output shape is: image_id -> list of tokenized reference captions.
    """
    references = defaultdict(list)

    for batch in tqdm(loader, desc="build references", leave=False):
        _, _, _, image_ids, raw_captions = batch

        for image_id, raw_caption in zip(image_ids, raw_captions):
            references[image_id].append(tokenize(raw_caption))

    return dict(references)


@torch.no_grad()
def generate_batch_token_ids(
    model: nn.Module,
    images: torch.Tensor,
    cfg,
) -> torch.Tensor:
    """
    Generate token IDs for a batch of images using the model inference API.

    Expected model API:
        model.generate(images, max_len, bos_idx, eos_idx)
    """
    device = resolve_device(cfg)

    model.eval()

    generated = model.generate(
        images.to(device, non_blocking=cfg.train.non_blocking),
        max_len=cfg.generation.max_len,
        bos_idx=cfg.model.bos_idx,
        eos_idx=cfg.model.eos_idx,
    )

    if isinstance(generated, tuple):
        # Some models may return (token_ids, extra_info), e.g. attention maps.
        generated = generated[0]

    if not isinstance(generated, torch.Tensor):
        generated = torch.as_tensor(generated)

    return generated.detach().cpu()


# =========================================================
# WANDB HELPERS
# =========================================================
def _wandb_safe(obj):
    """Convert config objects into values W&B can serialize as JSON."""
    if is_dataclass(obj):
        return _wandb_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _wandb_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_wandb_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_wandb_safe(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    if callable(obj) and hasattr(obj, "__name__"):
        return obj.__name__

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        # Last resort for objects such as transforms or callables.
        return repr(obj)

def _filter_wandb_metrics(metrics: dict, allowlist: set[str]) -> dict:
    """Return only selected W&B metrics, always keeping the epoch index."""
    if not allowlist:
        return metrics
    return {"epoch": metrics["epoch"], **{k: v for k, v in metrics.items() if k in allowlist}}


def _is_better(value: float, best: float | None, mode: str, cfg: ICExperimentConfig) -> bool:
    """Decide whether a metric improves enough to replace the current best."""
    if best is None:
        return True
    if mode == "min":
        return value < best - cfg.train.early_stopping_min_delta
    return value > best + cfg.train.early_stopping_min_delta

def save_checkpoint(path, filename, model, cfg, extra=None):
    """Save a portable checkpoint with model weights and model configuration."""

    model_cfg = cfg.model

    payload = {
        "model_name": getattr(model_cfg, "name", model.__class__.__name__),
        "run_name": getattr(cfg.wandb, "run_name", None),
        "model_config": asdict(model_cfg) if is_dataclass(model_cfg) else vars(model_cfg),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, Path(path) / filename)


def build_optimizer_and_scheduler(model: nn.Module, cfg):
    """Create optimizer and optional LR scheduler from experiment config."""
    optimizer = cfg.optimizer.cls(
        (p for p in model.parameters() if p.requires_grad),
        **cfg.optimizer.kwargs,
    )
    scheduler = (
        None
        if cfg.scheduler.cls is None
        else cfg.scheduler.cls(optimizer, **cfg.scheduler.kwargs)
    )
    return optimizer, scheduler


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count all model parameters and the subset that receives gradients."""
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def init_wandb_run(cfg, model: nn.Module | None = None, parameter_counts: dict[str, int] | None = None):
    """Start a W&B run and log optional model metadata."""
    if not cfg.wandb.enabled or cfg.wandb.mode == "disabled":
        return None

    import wandb

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        tags=cfg.wandb.tags,
        notes=cfg.wandb.notes,
        mode=cfg.wandb.mode,
        config=_wandb_safe(cfg),
    )

    wandb.define_metric("epoch")
    for metric_name in cfg.wandb.metric_allowlist:
        # Explicitly tie allowed metrics to epoch for cleaner W&B charts.
        wandb.define_metric(metric_name, step_metric="epoch")

    if model is not None and cfg.wandb.watch_model:
        wandb.watch(model, log=cfg.wandb.watch_log, log_freq=cfg.wandb.watch_log_freq)

    if parameter_counts is not None:
        wandb.log(
            {
                "model/total_params": parameter_counts["total"],
                "model/trainable_params": parameter_counts["trainable"],
            }
        )

    return run


def log_wandb_summary(wandb_run, cfg, summary_values: dict[str, Any]) -> None:
    """Write final/best metrics to the W&B run summary."""
    if wandb_run is None:
        return

    for key, value in summary_values.items():
        if value is None:
            continue
        if cfg.wandb.summary_allowlist and key not in cfg.wandb.summary_allowlist:
            continue
        wandb_run.summary[key] = value


def build_generated_examples_dataframe(
    model: nn.Module,
    fixed_eval_subset: dict[str, Any],
    tokenizer,
    cfg,
) -> Any:
    """Build a table of reference and generated captions for inspection."""
    import pandas as pd

    generated_ids = generate_batch_token_ids(
        model=model,
        images=fixed_eval_subset["images"],
        cfg=cfg,
    )

    examples = []
    for image_id, reference, pred_ids in zip(
        fixed_eval_subset["image_ids"],
        fixed_eval_subset["raw_captions"],
        generated_ids,
    ):
        examples.append(
            {
                "image_id": image_id,
                "reference_caption": reference,
                "generated_caption": decode_token_ids(
                    pred_ids,
                    tokenizer=tokenizer,
                    eos_idx=cfg.model.eos_idx,
                    skip_special_tokens=True,
                ),
            }
        )

    return pd.DataFrame(examples)


def run_captioning_experiment(
    *,
    model: nn.Module,
    cfg,
    train_loader,
    val_loader,
    image_loader,
    references_by_image_id: dict[str, list[list[str]]],
    fixed_eval_subset: dict[str, Any],
    tokenizer,
    model_label: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    save_dir: str | Path = "saved_models",
    checkpoint_dir: str | Path = "checkpoints",
    restore_best: bool = True,
    log_to_wandb: bool = True,
) -> dict[str, Any]:
    """
    Shared train/eval/checkpoint/W&B loop for both captioning models.

    The model only needs the common notebook API:
    ``model(images, decoder_inputs)`` and ``model.generate(...)``.
    """
    import pandas as pd

    optimizer = optimizer or build_optimizer_and_scheduler(model, cfg)[0]
    parameter_counts = count_parameters(model)
    wandb_run = init_wandb_run(
        cfg,
        model=model,
        parameter_counts=parameter_counts,
    ) if log_to_wandb else None

    history = []
    best_value = None
    best_epoch = None
    best_state = None
    best_metrics = None
    epochs_without_improvement = 0

    for epoch in range(1, cfg.train.epochs + 1):
        # Training and validation both use teacher forcing; BLEU uses generate().
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            desc=f"{model_label} train {epoch}/{cfg.train.epochs}",
        )
        val_metrics = evaluate_loss(
            model=model,
            loader=val_loader,
            cfg=cfg,
            desc=f"{model_label} eval {epoch}/{cfg.train.epochs}",
        )

        if epoch % cfg.evaluation.compute_bleu_every_n_epochs == 0:
            bleu_metrics = evaluate_bleu(
                model=model,
                image_loader=image_loader,
                references_by_image_id=references_by_image_id,
                tokenizer=tokenizer,
                cfg=cfg,
                max_batches=cfg.evaluation.max_bleu_batches,
            )
        else:
            bleu_metrics = {}

        metrics = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/perplexity": train_metrics["perplexity"],
            "val/loss": val_metrics["loss"],
            "val/perplexity": val_metrics["perplexity"],
            **{f"val/{k}": v for k, v in bleu_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
        }

        current_value = metrics[cfg.train.best_metric]
        if _is_better(current_value, best_value, cfg.train.best_mode, cfg):
            best_value = current_value
            best_epoch = epoch
            # Keep a CPU copy so the best model can be restored after training.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics.copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metrics["best/epoch_so_far"] = best_epoch
        metrics[f"best/{cfg.train.best_metric.replace('/', '_')}_so_far"] = best_value
        history.append(metrics)

        if scheduler is not None:
            # Config currently assumes metric-based schedulers such as ReduceLROnPlateau.
            scheduler_metric = metrics[cfg.scheduler.step_metric]
            scheduler.step(scheduler_metric)

        if wandb_run is not None and cfg.wandb.log_epoch_metrics:
            if epoch % cfg.wandb.log_every_n_epochs == 0:
                import wandb

                wandb.log(_filter_wandb_metrics(metrics, cfg.wandb.metric_allowlist))

        print(
            f"epoch={epoch:03d} | "
            f"train_loss={metrics['train/loss']:.4f} | "
            f"val_loss={metrics['val/loss']:.4f} | "
            f"val_ppl={metrics['val/perplexity']:.2f} | "
            f"bleu4={metrics.get('val/bleu_4', float('nan')):.4f}"
        )

        if cfg.train.early_stopping and epochs_without_improvement >= cfg.train.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}.")
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.wandb.run_name or model_label
    save_dir = Path(save_dir)
    checkpoint_dir = Path(checkpoint_dir)
    save_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_model_path = save_dir / f"{timestamp}_{run_name}_last.pt"
    best_model_path = save_dir / f"{timestamp}_{run_name}_best.pt"
    last_checkpoint_file = f"{run_name}_{timestamp}_last.pth"
    best_checkpoint_file = f"{run_name}_{timestamp}_best.pth"

    last_epoch = history[-1]["epoch"]
    last_metrics = history[-1]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "last_epoch": last_epoch,
            "last_metrics": last_metrics,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        },
        last_model_path,
    )
    save_checkpoint(
        path=checkpoint_dir,
        filename=last_checkpoint_file,
        model=model,
        cfg=cfg,
        extra={
            "epoch": last_epoch,
            "metrics": last_metrics,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "checkpoint_type": "last",
        },
    )

    if restore_best and best_state is not None:
        # Final evaluation and qualitative examples should use the best weights.
        model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
            },
            best_model_path,
        )
        save_checkpoint(
            path=checkpoint_dir,
            filename=best_checkpoint_file,
            model=model,
            cfg=cfg,
            extra={
                "epoch": best_epoch,
                "metrics": best_metrics,
                "checkpoint_type": "best",
            },
        )

    history_df = pd.DataFrame(history)
    final_val_metrics = evaluate_loss(model, val_loader, cfg, desc=f"{model_label} final eval")
    final_bleu_metrics = evaluate_bleu(
        model=model,
        image_loader=image_loader,
        references_by_image_id=references_by_image_id,
        tokenizer=tokenizer,
        cfg=cfg,
        max_batches=cfg.evaluation.max_bleu_batches,
    )
    examples_df = build_generated_examples_dataframe(
        model=model,
        fixed_eval_subset=fixed_eval_subset,
        tokenizer=tokenizer,
        cfg=cfg,
    )

    if wandb_run is not None:
        import wandb

        if cfg.wandb.log_generated_examples:
            wandb.log({f"qualitative/{model_label}_generated_captions": wandb.Table(dataframe=examples_df)})

        log_wandb_summary(
            wandb_run,
            cfg,
            {
                "best/epoch": best_epoch,
                "best/val/loss": best_metrics.get("val/loss") if best_metrics else None,
                "best/val/perplexity": best_metrics.get("val/perplexity") if best_metrics else None,
                "best/val/bleu_1": best_metrics.get("val/bleu_1") if best_metrics else None,
                "best/val/bleu_2": best_metrics.get("val/bleu_2") if best_metrics else None,
                "best/val/bleu_3": best_metrics.get("val/bleu_3") if best_metrics else None,
                "best/val/bleu_4": best_metrics.get("val/bleu_4") if best_metrics else None,
                "final/val/loss": final_val_metrics["loss"],
                "final/val/perplexity": final_val_metrics["perplexity"],
                "final/val/bleu_1": final_bleu_metrics["bleu_1"],
                "final/val/bleu_2": final_bleu_metrics["bleu_2"],
                "final/val/bleu_3": final_bleu_metrics["bleu_3"],
                "final/val/bleu_4": final_bleu_metrics["bleu_4"],
                "model/total_params": parameter_counts["total"],
                "model/trainable_params": parameter_counts["trainable"],
            },
        )
        wandb.save(str(last_model_path))
        wandb.save(str(checkpoint_dir / last_checkpoint_file))
        if restore_best and best_state is not None:
            wandb.save(str(best_model_path))
            wandb.save(str(checkpoint_dir / best_checkpoint_file))
        wandb.finish()

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "history": history,
        "history_df": history_df,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_val_metrics": final_val_metrics,
        "final_bleu_metrics": final_bleu_metrics,
        "examples_df": examples_df,
        "last_model_path": last_model_path,
        "best_model_path": best_model_path if restore_best and best_state is not None else None,
        "last_checkpoint_path": checkpoint_dir / last_checkpoint_file,
        "best_checkpoint_path": checkpoint_dir / best_checkpoint_file if restore_best and best_state is not None else None,
        "parameter_counts": parameter_counts,
    }
