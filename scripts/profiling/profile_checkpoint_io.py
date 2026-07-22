"""Bounded, disposable checkpoint-I/O profiling for VASU models.

All output is written beneath ``tmp/profiling/checkpoints`` and removed unless
``--retain`` is supplied.  No production checkpoint is read or overwritten.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator

import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.config import TrainConfig, get_vasu_60m_config  # noqa: E402
from vasu.model.model import VASUModel  # noqa: E402
from vasu.training.checkpoint import save_checkpoint  # noqa: E402
from vasu.training.optimizer import build_optimizer  # noqa: E402
from vasu.training.resume_state import capture_gradient_state, capture_rng_state  # noqa: E402


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _progress(gradients: dict[str, torch.Tensor], scaler: torch.amp.GradScaler) -> dict[str, Any]:
    return {
        "format_version": 2,
        "sampler": {
            "format_version": 1,
            "num_samples": 8,
            "batch_size": 2,
            "shuffle": True,
            "seed": 42,
            "drop_last": True,
            "epoch": 0,
            "next_batch_index": 1,
        },
        "phase": "train",
        "accumulated_microbatches": 1 if gradients else 0,
        "optimizer_steps_in_epoch": 1,
        "gradients": gradients,
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
    }


@contextmanager
def _instrument_save() -> Iterator[dict[str, float]]:
    """Measure actual ``save_checkpoint`` serialization and atomic rename."""

    import vasu.training.checkpoint as checkpoint_module

    timings: dict[str, float] = {"serialize_seconds": 0.0, "replace_seconds": 0.0}
    original_save = checkpoint_module.torch.save
    original_replace = checkpoint_module.os.replace

    def timed_save(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_save(*args, **kwargs)
        finally:
            timings["serialize_seconds"] += time.perf_counter() - started

    def timed_replace(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_replace(*args, **kwargs)
        finally:
            timings["replace_seconds"] += time.perf_counter() - started

    checkpoint_module.torch.save = timed_save
    checkpoint_module.os.replace = timed_replace
    try:
        yield timings
    finally:
        checkpoint_module.torch.save = original_save
        checkpoint_module.os.replace = original_replace


def _one_optimizer_step(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> None:
    inputs = torch.randint(0, model.config.vocab_size, (2, 256), device=device)
    targets = torch.randint_like(inputs, high=model.config.vocab_size)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        loss = functional.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def _save_case(
    *,
    name: str,
    destination: Path,
    model: VASUModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    gradients: dict[str, torch.Tensor],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if optimizer is not None:
        metadata["best_val_loss"] = 1.0
        metadata["training_progress"] = _progress(gradients, scaler)
    with _instrument_save() as timings:
        started = time.perf_counter()
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            loss=1.0,
            path=destination,
            global_step=1,
            **metadata,
        )
        total_seconds = time.perf_counter() - started
    started = time.perf_counter()
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    load_seconds = time.perf_counter() - started
    if destination.with_suffix(destination.suffix + ".tmp").exists():
        raise RuntimeError("atomic checkpoint save left an orphan temporary file")
    return {
        "case": name,
        "path": str(destination),
        "total_save_seconds": total_seconds,
        "serialization_seconds": timings["serialize_seconds"],
        "atomic_replace_seconds": timings["replace_seconds"],
        "load_seconds": load_seconds,
        "file_bytes": destination.stat().st_size,
        "optimizer_tensor_bytes": _tensor_bytes(payload.get("optimizer", {})),
        "gradient_tensor_bytes": _tensor_bytes(
            payload.get("training_progress", {}).get("gradients", {})
        ),
        "metadata_tensor_bytes": _tensor_bytes(
            payload.get("training_progress", {})
        ),
        "keys": sorted(payload),
    }


def profile_checkpoint_io(
    *,
    directory: Path,
    backend: str,
    retain: bool,
) -> list[dict[str, Any]]:
    """Profile model-only, boundary, and partial-accumulation checkpoints."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "fused" and device.type != "cuda":
        raise RuntimeError("fused checkpoint profile requires CUDA.")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    model = VASUModel(get_vasu_60m_config()).to(device).train()
    config = TrainConfig(learning_rate=1e-4, weight_decay=0.1, optimizer_backend=backend)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    _one_optimizer_step(model, optimizer, scaler, device)
    boundary = _save_case(
        name="optimizer_boundary",
        destination=directory / "boundary.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradients={},
    )
    inputs = torch.randint(0, model.config.vocab_size, (2, 256), device=device)
    targets = torch.randint_like(inputs, high=model.config.vocab_size)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        loss = functional.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
    scaler.scale(loss / 2).backward()
    gradients = capture_gradient_state(model)
    mid = _save_case(
        name="mid_accumulation",
        destination=directory / "mid.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        gradients=gradients,
    )
    model_only = _save_case(
        name="model_only",
        destination=directory / "model_only.pt",
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=scaler,
        gradients={},
    )
    _sync(device)
    results = [boundary, mid, model_only]
    if not retain:
        shutil.rmtree(directory)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("standard", "fused"), default="standard")
    parser.add_argument("--directory", type=Path, default=Path("tmp/profiling/checkpoints"))
    parser.add_argument("--retain", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "backend": args.backend,
        "retained": args.retain,
        "results": profile_checkpoint_io(
            directory=args.directory, backend=args.backend, retain=args.retain
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
