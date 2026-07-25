from __future__ import annotations

from os import PathLike
import os
from pathlib import Path
from typing import Any

import torch


def _is_path(value: object) -> bool:
    return isinstance(value, (str, PathLike))


def save_checkpoint(
    model,
    optimizer,
    scheduler=None,
    epoch=0,
    loss=None,
    path=None,
    global_step=0,
    verify_after_write: bool = False,
    fsync: bool = False,
    **metadata: Any,
) -> None:
    """Save a training checkpoint while preserving the legacy call format.

    Supported forms:

        save_checkpoint(model, optimizer, epoch, loss, path)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            loss=loss,
            path=path,
            global_step=global_step,
        )
    """

    # Compatibility with the original positional API:
    # save_checkpoint(model, optimizer, epoch, loss, path[, global_step])
    if _is_path(loss) and (path is None or not _is_path(path)):
        legacy_global_step = path
        path = loss
        loss = epoch
        epoch = scheduler
        scheduler = None

        if legacy_global_step is not None:
            global_step = legacy_global_step

    if path is None or not _is_path(path):
        raise TypeError("path must be a string or path-like object")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": model.state_dict(),
        "loss": loss,
    }

    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    overlap = payload.keys() & metadata.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(
            f"metadata cannot replace reserved checkpoint keys: {names}"
        )

    payload.update(metadata)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        torch.save(payload, temporary)
        if fsync:
            # Windows requires a writable file descriptor for ``fsync``.
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
        if verify_after_write:
            restored = torch.load(
                temporary,
                map_location="cpu",
                mmap=True,
                weights_only=False,
            )
            if not isinstance(restored, dict):
                raise ValueError("temporary checkpoint payload is not a mapping")
            missing = payload.keys() - restored.keys()
            if missing:
                raise ValueError(
                    "temporary checkpoint is missing keys: "
                    + ", ".join(sorted(missing))
                )
            if int(restored["global_step"]) != int(global_step):
                raise ValueError("temporary checkpoint global_step mismatch")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    *,
    map_location="cpu",
    strict: bool = True,
) -> int:
    """Restore available state and return the next epoch.

    Older checkpoints without scheduler or global-step state remain valid.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=strict,
    )

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = int(checkpoint.get("epoch", -1))
    return epoch + 1
