from __future__ import annotations

from typing import Callable
import torch.nn as nn

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    def decorator(cls: type[nn.Module]) -> type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model name '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(model_cfg) -> nn.Module:
    if model_cfg.name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model name: {model_cfg.name}. Available models: {available}"
        )
    model_cls = MODEL_REGISTRY[model_cfg.name]
    return model_cls(**model_cfg.kwargs)

