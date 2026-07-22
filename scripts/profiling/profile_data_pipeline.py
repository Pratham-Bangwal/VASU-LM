"""Bounded, Windows-safe profiling for VASU local token data pipelines.

The tool reads existing datasets only.  The optional synthetic-model step uses
ephemeral model/optimizer state and never writes production checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.config import ModelConfig, get_vasu_60m_config  # noqa: E402
from vasu.model.model import VASUModel  # noqa: E402
from vasu.training.dataset import TextDataset  # noqa: E402
from vasu.training.instruction_dataset import PackedInstructionDataset  # noqa: E402


DatasetKind = Literal["text", "packed"]


@dataclass(frozen=True)
class DatasetSpec:
    """A local, non-mutating dataset profile target."""

    name: str
    kind: DatasetKind
    token_path: Path
    mask_path: Path | None = None


DEFAULT_SPECS = (
    DatasetSpec("fineweb_100k", "text", Path("data/processed/pretrain/fineweb_100k.bin")),
    DatasetSpec(
        "factual_cpt_mixture",
        "text",
        Path("data/processed/pretrain/mixtures/vasu_60m_fineweb_wikimedia_85_15.bin"),
    ),
    DatasetSpec(
        "alpaca_masked_v2",
        "packed",
        Path("data/processed/instruct/alpaca_masked_v2.bin"),
        Path("data/processed/instruct/alpaca_masked_v2_mask.bin"),
    ),
    DatasetSpec(
        "ultrachat_masked_v2",
        "packed",
        Path("data/processed/instruct/ultrachat_masked_v2.bin"),
        Path("data/processed/instruct/ultrachat_masked_v2_mask.bin"),
    ),
)


def available_datasets() -> dict[str, DatasetSpec]:
    """Return only complete, known local profile targets."""

    return {
        spec.name: spec
        for spec in DEFAULT_SPECS
        if spec.token_path.is_file()
        and (spec.mask_path is None or spec.mask_path.is_file())
    }


def validate_loader_options(
    *,
    workers: int,
    prefetch_factor: int | None,
) -> None:
    """Reject DataLoader settings that PyTorch cannot apply safely."""

    if workers < 0:
        raise ValueError("workers must be non-negative.")
    if prefetch_factor is not None and prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive when supplied.")
    if workers == 0 and prefetch_factor is not None:
        raise ValueError(
            "prefetch_factor requires workers > 0; no worker process exists "
            "when workers=0."
        )


def build_dataset(spec: DatasetSpec, sequence_length: int) -> Dataset[Any]:
    """Open an existing memmap-backed dataset without changing it."""

    if spec.kind == "text":
        return TextDataset(str(spec.token_path), seq_len=sequence_length)
    if spec.mask_path is None:
        raise ValueError(f"Packed dataset {spec.name!r} has no mask path.")
    return PackedInstructionDataset(
        str(spec.token_path), str(spec.mask_path), sequence_length
    )


def build_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
) -> DataLoader[Any]:
    """Build a non-persistent worker loader suitable for Windows profiling."""

    validate_loader_options(workers=workers, prefetch_factor=prefetch_factor)
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": True,
        "pin_memory": pin_memory,
        "num_workers": workers,
        "persistent_workers": False,
    }
    if workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory_mib(device: torch.device) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    mib = 1024**2
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / mib,
        "reserved_mib": torch.cuda.memory_reserved(device) / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / mib,
    }


def _batch_to_device(
    batch: Any,
    device: torch.device,
    *,
    non_blocking: bool,
) -> tuple[list[torch.Tensor], float]:
    _sync(device)
    started = time.perf_counter()
    tensors = [item.to(device, non_blocking=non_blocking) for item in batch]
    _sync(device)
    return tensors, time.perf_counter() - started


def _summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "spread": max(values) - min(values),
    }


def profile_loader(
    *,
    spec: DatasetSpec,
    sequence_length: int,
    batch_size: int,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    warmup: int,
    iterations: int,
    device: torch.device,
    include_step: bool,
    model_name: str,
) -> dict[str, Any]:
    """Measure local loading, transfer, and optional ephemeral GPU steps."""

    dataset = build_dataset(spec, sequence_length)
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    iterator = iter(loader)
    started = time.perf_counter()
    first_batch = next(iterator)
    first_batch_seconds = time.perf_counter() - started
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)

    model: VASUModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    scaler: torch.amp.GradScaler | None = None
    if include_step:
        config = ModelConfig() if model_name == "31m" else get_vasu_60m_config()
        model = VASUModel(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    loader_seconds: list[float] = []
    transfer_blocking_seconds: list[float] = []
    transfer_non_blocking_seconds: list[float] = []
    compute_seconds: list[float] = []
    full_seconds: list[float] = []
    batch = first_batch
    for index in range(iterations):
        if index:
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            loader_seconds.append(time.perf_counter() - started)
        _, blocking = _batch_to_device(batch, device, non_blocking=False)
        transfer_blocking_seconds.append(blocking)
        _, non_blocking = _batch_to_device(
            batch, device, non_blocking=pin_memory and device.type == "cuda"
        )
        transfer_non_blocking_seconds.append(non_blocking)
        if model is not None and optimizer is not None and scaler is not None:
            tensors, transfer = _batch_to_device(
                batch, device, non_blocking=pin_memory and device.type == "cuda"
            )
            inputs, targets = tensors[:2]
            optimizer.zero_grad(set_to_none=True)
            _sync(device)
            compute_started = time.perf_counter()
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                loss = functional.cross_entropy(
                    model(inputs).flatten(0, 1), targets.flatten()
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _sync(device)
            compute = time.perf_counter() - compute_started
            compute_seconds.append(compute)
            full_seconds.append(transfer + compute)

    samples = batch_size * sequence_length
    report: dict[str, Any] = {
        "dataset": spec.name,
        "dataset_kind": spec.kind,
        "token_path": str(spec.token_path),
        "token_file_bytes": spec.token_path.stat().st_size,
        "mask_file_bytes": (
            spec.mask_path.stat().st_size if spec.mask_path is not None else None
        ),
        "dataset_samples": len(dataset),
        "item_structure": 3 if spec.kind == "packed" else 2,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "workers": workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor,
        "first_batch_seconds": first_batch_seconds,
        "loader_steady_seconds": _summarize(loader_seconds) if loader_seconds else None,
        "transfer_blocking_seconds": _summarize(transfer_blocking_seconds),
        "transfer_non_blocking_seconds": _summarize(transfer_non_blocking_seconds),
        "cuda_memory": _memory_mib(device),
    }
    if loader_seconds:
        report["loader_batches_per_second"] = len(loader_seconds) / sum(loader_seconds)
        report["loader_tokens_per_second"] = (
            samples * report["loader_batches_per_second"]
        )
    if full_seconds:
        report["end_to_end_step_seconds"] = _summarize(full_seconds)
        report["end_to_end_tokens_per_second"] = samples / statistics.median(full_seconds)
        report["compute_seconds"] = _summarize(compute_seconds)
        report["loader_wait_fraction"] = (
            statistics.median(loader_seconds) / statistics.median(full_seconds)
            if loader_seconds
            else None
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(available_datasets()), default="fineweb_100k")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--include-step", action="store_true")
    parser.add_argument("--model", choices=("31m", "60m"), default="60m")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Run only from an explicit script entry point for Windows safety."""

    args = parse_args()
    if min(args.batch_size, args.sequence_length, args.warmup, args.iterations) < 1:
        raise ValueError("batch size, sequence length, warmup, and iterations must be positive.")
    validate_loader_options(workers=args.workers, prefetch_factor=args.prefetch_factor)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    report = profile_loader(
        spec=available_datasets()[args.dataset],
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
        include_step=args.include_step,
        model_name=args.model,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
