from typing import Any

import torch
import torch.nn as nn
from vasu.config import TrainConfig


def build_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    """Build AdamW with an explicit, backward-compatible backend choice."""

    backend = getattr(config, "optimizer_backend", "standard")
    if backend not in {"auto", "standard", "foreach", "fused"}:
        raise ValueError(
            "optimizer_backend must be one of: auto, standard, foreach, fused"
        )
    if backend == "auto":
        backend = "standard"
    device = next(model.parameters()).device
    kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if backend == "foreach":
        kwargs["foreach"] = True
    elif backend == "fused":
        if device.type != "cuda":
            raise ValueError("fused AdamW requires CUDA parameters")
        kwargs["fused"] = True

    return torch.optim.AdamW(
        model.parameters(),
        **kwargs,
    )
