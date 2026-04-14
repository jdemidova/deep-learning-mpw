from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_grad_cam import (
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    HiResCAM,
    LayerCAM,
    ScoreCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


CAM_METHODS = {
    "gradcam": GradCAM,
    "hirescam": HiResCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "layercam": LayerCAM,
    "eigengradcam": EigenGradCAM,
}


def _base_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _to_int(x: Any) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.detach().cpu().item())
    return int(x)


def _unpack_sample(sample: Any) -> tuple[torch.Tensor, int]:
    """
    Supports:
      - (image, label)
      - (image, label, ...)
      - {"image": ..., "label": ...}
    """
    if isinstance(sample, dict):
        return sample["image"], _to_int(sample["label"])

    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        return sample[0], _to_int(sample[1])

    raise TypeError(
        "Unsupported dataset sample format. Expected tuple/list with (image, label) "
        "or dict with keys 'image' and 'label'."
    )


def _find_last_conv_layer(model: nn.Module) -> nn.Module:
    base = _base_model(model)
    conv_layers = [m for m in base.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No nn.Conv2d layer found. Grad-CAM needs a conv-like target layer.")
    return conv_layers[-1]


def resolve_target_layers(
    model: nn.Module,
    target_layers: Sequence[nn.Module] | None = None,
) -> list[nn.Module]:
    if target_layers is not None:
        return list(target_layers)
    return [_find_last_conv_layer(model)]


def denorm_to_rgb(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """
    image_tensor: CHW normalized tensor
    returns: HWC float32 in [0, 1]
    """
    x = image_tensor.detach().cpu().float()
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image tensor, got shape={tuple(x.shape)}")

    mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    x = x * std_t + mean_t
    x = x.clamp(0, 1)

    # display helper for grayscale inputs
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)

    return x.permute(1, 2, 0).numpy()


def _get_device(model: nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.detach().cpu(), dim=0)


def compute_cam_for_tensor(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None,
    *,
    target_layers: Sequence[nn.Module] | None = None,
    method: str = "gradcam",
    device: str | torch.device | None = None,
    aug_smooth: bool = False,
    eigen_smooth: bool = True,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    image_tensor: CHW or 1xCHW normalized tensor
    target_class:
      - int => CAM for that class
      - None => CAM for predicted class
    returns:
      grayscale_cam (H, W), logits (num_classes,)
    """
    model.eval()
    dev = _get_device(model, device)
    layers = resolve_target_layers(model, target_layers)

    x = image_tensor
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(dev)

    method_key = method.lower()
    if method_key not in CAM_METHODS:
        raise ValueError(f"Unknown CAM method '{method}'. Available: {sorted(CAM_METHODS)}")

    cam_cls = CAM_METHODS[method_key]
    targets = None if target_class is None else [ClassifierOutputTarget(int(target_class))]

    with torch.enable_grad():
        with cam_cls(model=model, target_layers=layers) as cam:
            grayscale_cam = cam(
                input_tensor=x,
                targets=targets,
                aug_smooth=aug_smooth,
                eigen_smooth=eigen_smooth,
            )[0]
            logits = cam.outputs.detach().cpu()[0]

    return grayscale_cam, logits


def show_heatmap_on_image(
    model: nn.Module,
    dataset,
    index: int,
    target_class: int | None,
    *,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    target_layers: Sequence[nn.Module] | None = None,
    method: str = "gradcam",
    device: str | torch.device | None = None,
    aug_smooth: bool = False,
    eigen_smooth: bool = True,
    figsize: tuple[int, int] = (12, 4),
) -> dict[str, Any]:
    """
    Show heatmap on chosen image + chosen class
    """
    image_tensor, true_class = _unpack_sample(dataset[index])

    grayscale_cam, logits = compute_cam_for_tensor(
        model=model,
        image_tensor=image_tensor,
        target_class=target_class,
        target_layers=target_layers,
        method=method,
        device=device,
        aug_smooth=aug_smooth,
        eigen_smooth=eigen_smooth,
    )

    probs = _softmax_probs(logits)
    pred_class = int(torch.argmax(probs).item())
    effective_target = pred_class if target_class is None else int(target_class)

    rgb_img = denorm_to_rgb(image_tensor, mean, std)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(rgb_img)
    axes[0].set_title(
        f"image idx={index}\ntrue={class_names[true_class]}\npred={class_names[pred_class]}"
    )
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title(
        f"raw CAM\ntarget={class_names[effective_target]}\np={probs[effective_target]:.3f}"
    )
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"{method} overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "index": index,
        "true_class": true_class,
        "pred_class": pred_class,
        "target_class": effective_target,
        "logits": logits,
        "probs": probs,
        "cam": grayscale_cam,
        "overlay": overlay,
    }


def show_heatmaps_for_all_classes_on_image(
    model: nn.Module,
    dataset,
    index: int,
    *,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    target_layers: Sequence[nn.Module] | None = None,
    method: str = "gradcam",
    device: str | torch.device | None = None,
    aug_smooth: bool = False,
    eigen_smooth: bool = True,
    ncols: int = 4,
    figsize_per_cell: tuple[float, float] = (4.0, 4.0),
) -> dict[str, Any]:
    """
    2) chosen image + CAMs for all classes
    """
    image_tensor, true_class = _unpack_sample(dataset[index])
    rgb_img = denorm_to_rgb(image_tensor, mean, std)

    layers = resolve_target_layers(model, target_layers)
    dev = _get_device(model, device)
    model.eval()

    x = image_tensor.unsqueeze(0).to(dev)
    method_key = method.lower()
    cam_cls = CAM_METHODS[method_key]

    overlays = []
    raw_cams = []
    logits = None

    with torch.enable_grad():
        with cam_cls(model=model, target_layers=layers) as cam:
            for class_idx in range(len(class_names)):
                grayscale_cam = cam(
                    input_tensor=x,
                    targets=[ClassifierOutputTarget(class_idx)],
                    aug_smooth=aug_smooth,
                    eigen_smooth=eigen_smooth,
                )[0]
                if logits is None:
                    logits = cam.outputs.detach().cpu()[0]

                raw_cams.append(grayscale_cam)
                overlays.append(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))

    assert logits is not None
    probs = _softmax_probs(logits)
    pred_class = int(torch.argmax(probs).item())

    total_panels = 1 + len(class_names)
    ncols = min(ncols, total_panels)
    nrows = math.ceil(total_panels / ncols)
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(rgb_img)
    axes[0].set_title(
        f"image idx={index}\ntrue={class_names[true_class]}\npred={class_names[pred_class]}"
    )
    axes[0].axis("off")

    for class_idx, ax in enumerate(axes[1 : 1 + len(class_names)]):
        ax.imshow(overlays[class_idx])
        ax.set_title(
            f"target={class_names[class_idx]}\np={probs[class_idx]:.3f}"
        )
        ax.axis("off")

    for ax in axes[1 + len(class_names) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "index": index,
        "true_class": true_class,
        "pred_class": pred_class,
        "logits": logits,
        "probs": probs,
        "cams": raw_cams,
        "overlays": overlays,
    }


def sample_one_index_per_true_class(
    dataset,
    *,
    num_classes: int,
    seed: int = 0,
) -> dict[int, int]:
    buckets: dict[int, list[int]] = defaultdict(list)

    for idx in range(len(dataset)):
        _, y = _unpack_sample(dataset[idx])
        buckets[y].append(idx)

    rng = random.Random(seed)
    selected: dict[int, int] = {}

    for class_idx in range(num_classes):
        indices = buckets.get(class_idx, [])
        if indices:
            selected[class_idx] = rng.choice(indices)

    return selected


def show_target_class_across_true_classes(
    model: nn.Module,
    dataset,
    target_class: int,
    *,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    seed: int = 0,
    target_layers: Sequence[nn.Module] | None = None,
    method: str = "gradcam",
    device: str | torch.device | None = None,
    aug_smooth: bool = False,
    eigen_smooth: bool = True,
    ncols: int = 4,
    figsize_per_cell: tuple[float, float] = (4.0, 4.0),
) -> dict[str, Any]:
    """
    3) fixed target class + one random VAL image from each true class
    """
    selected = sample_one_index_per_true_class(
        dataset,
        num_classes=len(class_names),
        seed=seed,
    )

    items = []
    for true_class, idx in selected.items():
        image_tensor, _ = _unpack_sample(dataset[idx])
        grayscale_cam, logits = compute_cam_for_tensor(
            model=model,
            image_tensor=image_tensor,
            target_class=target_class,
            target_layers=target_layers,
            method=method,
            device=device,
            aug_smooth=aug_smooth,
            eigen_smooth=eigen_smooth,
        )
        probs = _softmax_probs(logits)
        pred_class = int(torch.argmax(probs).item())
        rgb_img = denorm_to_rgb(image_tensor, mean, std)
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        items.append(
            {
                "true_class": true_class,
                "index": idx,
                "pred_class": pred_class,
                "target_prob": float(probs[target_class].item()),
                "rgb_img": rgb_img,
                "cam": grayscale_cam,
                "overlay": overlay,
                "probs": probs,
                "logits": logits,
            }
        )

    total = len(items)
    ncols = min(ncols, total)
    nrows = math.ceil(total / ncols)
    fig_w = figsize_per_cell[0] * ncols
    fig_h = figsize_per_cell[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for ax, item in zip(axes, items):
        ax.imshow(item["overlay"])
        ax.set_title(
            "\n".join(
                [
                    f"true={class_names[item['true_class']]}",
                    f"pred={class_names[item['pred_class']]}",
                    f"target={class_names[target_class]}",
                    f"p(target)={item['target_prob']:.3f}",
                    f"idx={item['index']}",
                ]
            )
        )
        ax.axis("off")

    for ax in axes[len(items) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "target_class": target_class,
        "seed": seed,
        "selected_indices": {item["true_class"]: item["index"] for item in items},
        "items": items,
    }