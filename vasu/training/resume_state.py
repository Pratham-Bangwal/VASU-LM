"""Checkpointable runtime state required for exact mid-epoch continuation."""

from __future__ import annotations

import random
from typing import Any, Iterable

import numpy as np
import torch


FORMAT_VERSION = 2
RESUME_PHASES = frozenset({"train", "post_train_pre_validation", "next_epoch"})


def capture_gradient_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy currently accumulated gradients to CPU for checkpoint storage."""

    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def restore_gradient_state(
    model: torch.nn.Module,
    gradients: dict[str, torch.Tensor],
) -> None:
    """Restore accumulated gradients, rejecting stale or incomplete state."""

    parameters = dict(model.named_parameters())
    unexpected = set(gradients) - set(parameters)
    if unexpected:
        raise ValueError(
            "Checkpoint has gradients for unknown parameters: "
            f"{sorted(unexpected)[:5]}"
        )
    for parameter in parameters.values():
        parameter.grad = None
    for name, gradient in gradients.items():
        parameter = parameters[name]
        if gradient.shape != parameter.shape:
            raise ValueError(f"Gradient shape mismatch for parameter {name!r}.")
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNGs used by deterministic single-process training."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore all captured process RNGs exactly."""

    required = {"python", "numpy", "torch"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Checkpoint RNG state is incomplete: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # ``torch.load(..., map_location="cuda")`` also moves this CPU-generator
    # state to CUDA.  Both generator APIs require CPU ByteTensors, so normalize
    # device placement without changing the serialized checkpoint payload.
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable.")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def build_training_progress(
    *,
    sampler_state: dict[str, Any],
    phase: str,
    accumulated_microbatches: int,
    optimizer_steps_in_epoch: int,
    gradients: dict[str, torch.Tensor],
    scaler_state: dict[str, Any],
) -> dict[str, Any]:
    """Build additive checkpoint metadata for exact continuation."""

    if phase not in RESUME_PHASES:
        raise ValueError(f"Unsupported training resume phase: {phase!r}")
    if accumulated_microbatches < 0:
        raise ValueError("accumulated_microbatches must be non-negative.")
    if optimizer_steps_in_epoch < 0:
        raise ValueError("optimizer_steps_in_epoch must be non-negative.")
    return {
        "format_version": FORMAT_VERSION,
        "sampler": sampler_state,
        "phase": phase,
        "accumulated_microbatches": accumulated_microbatches,
        "optimizer_steps_in_epoch": optimizer_steps_in_epoch,
        "gradients": gradients,
        "scaler": scaler_state,
        "rng": capture_rng_state(),
    }


def validate_gradient_presence(
    gradients: dict[str, torch.Tensor],
    accumulated_microbatches: int,
) -> None:
    """Reject a mid-accumulation checkpoint that cannot resume exactly."""

    if accumulated_microbatches > 0 and not gradients:
        raise ValueError(
            "Checkpoint is mid-gradient-accumulation but has no gradients; "
            "exact resume is impossible."
        )


def finite_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Return whether all populated gradients are finite."""

    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )
