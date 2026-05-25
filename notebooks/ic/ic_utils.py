from __future__ import annotations

# stdlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, is_dataclass
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
    """Make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Mostly relevant for CUDA, MPS can still have some nondeterminism.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(cfg=None) -> torch.device:
    """Resolve device from config, with auto fallback."""
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
    Move tensor parts of dataset batch to device, keep metadata unchanged
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
    """Keep tokens only up to and including <eos>, if present."""
    if eos_idx in token_ids:
        return token_ids[: token_ids.index(eos_idx) + 1]
    return token_ids


# =========================================================
# TRAIN HELPERS
# =========================================================

def split_caption_inputs_targets(captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shift captions for teacher-forced next-token prediction.
    Task: teacher forcing / shifted next-token prediction.

    caption:  <bos> a dog runs <eos>
    input:    <bos> a dog runs
    target:        a dog runs <eos>
    """
    return captions[:, :-1].contiguous(), captions[:, 1:].contiguous()


def extract_logits(model_output: Any) -> torch.Tensor:
    """
    Support both:
      logits
      (logits, extra)
      {"logits": logits, ...}

    Task: allows models to share the same training loop.
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    if isinstance(model_output, dict):
        return model_output["logits"]
    return model_output

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    cfg,
    desc: str = "train",
) -> dict[str, float]:
    """One teacher-forced training epoch."""
    device = resolve_device(cfg)
    pad_idx = cfg.model.pad_idx
    grad_clip_norm = cfg.train.grad_clip_norm
    non_blocking = cfg.train.non_blocking

    model.train()

    total_loss_sum = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        images, captions, lengths, image_ids, raw_captions = move_batch_to_device(
            batch,
            device=device,
            non_blocking=non_blocking,
        )

        decoder_inputs, targets = split_caption_inputs_targets(captions)

        optimizer.zero_grad(set_to_none=True)

        model_output = model(images, decoder_inputs)
        loss, num_tokens = compute_captioning_loss(
            model_output,
            targets,
            pad_idx=pad_idx,
        )

        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

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
    Decode generated token IDs into readable text.
    Task: requires separate generate() inference path
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
    """Convert generated token IDs to tokenized text for BLEU"""
    decoded = decode_token_ids(
        token_ids,
        tokenizer=tokenizer,
        eos_idx=eos_idx,
        skip_special_tokens=True,
    )
    return tokenize(decoded)

def perplexity_from_loss(loss: float) -> float:
    """
    Perplexity = exp(mean_cross_entropy).
    Task: required perplexity metric.
    """
    return float(math.exp(min(loss, 20.0)))

def compute_captioning_loss(
    model_output: Any,
    targets: torch.Tensor,
    pad_idx: int,
) -> tuple[torch.Tensor, int]:
    """
    Cross-entropy over next-token predictions, ignoring <pad> targets.
    Task: masked CE loss, ignoring <pad>
    """
    logits = extract_logits(model_output)
    vocab_size = logits.size(-1)

    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )

    num_tokens = int((targets != pad_idx).sum().item())
    loss = loss_sum / max(num_tokens, 1)

    return loss, num_tokens

@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader,
    cfg,
    desc: str = "eval",
) -> dict[str, float]:
    """
    Teacher-forced evaluation loss/perplexity.
    Task: required perplexity metric
    """
    device = resolve_device(cfg)
    pad_idx = cfg.model.pad_idx
    non_blocking = cfg.train.non_blocking

    model.eval()

    total_loss_sum = 0.0
    total_tokens = 0

    progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        images, captions, lengths, image_ids, raw_captions = move_batch_to_device(
            batch,
            device=device,
            non_blocking=non_blocking,
        )

        decoder_inputs, targets = split_caption_inputs_targets(captions)

        model_output = model(images, decoder_inputs)
        loss, num_tokens = compute_captioning_loss(
            model_output,
            targets,
            pad_idx=pad_idx,
        )

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
    Corpus-level BLEU-1...BLEU-4.

    Generates one caption per image and compares it against all human references
    for the same image.

    Task: required BLEU-1...BLEU-4
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
            refs = references_by_image_id.get(image_id)

            if not refs:
                continue

            hyp = token_ids_to_bleu_tokens(
                pred_ids,
                tokenizer=tokenizer,
                eos_idx=eos_idx,
            )

            if len(hyp) == 0:
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

    Task: qualitative comparison of models
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

    Task: required BLEU-1...BLEU-4.
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
    Generate captions with model.generate().

    Expected model API:
        model.generate(images, max_len, bos_idx, eos_idx)

    Task: required separate generate() inference path
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
        generated = generated[0]

    if not isinstance(generated, torch.Tensor):
        generated = torch.as_tensor(generated)

    return generated.detach().cpu()


# =========================================================
# WANDB HELPERS
# =========================================================
def _wandb_safe(obj):
    """Helper for W&B config serialization."""
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
        return repr(obj)

def _filter_wandb_metrics(metrics: dict, allowlist: set[str]) -> dict:
    if not allowlist:
        return metrics
    return {"epoch": metrics["epoch"], **{k: v for k, v in metrics.items() if k in allowlist}}


def _is_better(value: float, best: float | None, mode: str, cfg: ICExperimentConfig) -> bool:
    if best is None:
        return True
    if mode == "min":
        return value < best - cfg.train.early_stopping_min_delta
    return value > best + cfg.train.early_stopping_min_delta

def save_checkpoint(path, filename, model, cfg, extra=None):

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