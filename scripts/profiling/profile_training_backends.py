"""Bounded synthetic AdamW backend benchmark for VASU CUDA execution."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
import time

import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.config import TrainConfig, get_vasu_60m_config  # noqa: E402
from vasu.model.model import VASUModel  # noqa: E402
from vasu.training.optimizer import build_optimizer  # noqa: E402


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_backend(backend: str, iterations: int) -> dict[str, float | str]:
    """Run warmup plus repeated AMP updates with deterministic synthetic data."""

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = VASUModel(get_vasu_60m_config()).to(device).train()
    config = TrainConfig(
        batch_size=2,
        learning_rate=1e-4,
        weight_decay=0.1,
        optimizer_backend=backend,
    )
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    inputs = torch.randint(32_000, (2, 256), device=device)
    targets = torch.randint_like(inputs, high=32_000)

    def step() -> tuple[float, float, float, float]:
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        begin = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=True):
            loss = functional.cross_entropy(
                model(inputs).flatten(0, 1), targets.flatten()
            )
        synchronize()
        forward = time.perf_counter() - begin
        begin = time.perf_counter()
        scaler.scale(loss).backward()
        synchronize()
        backward = time.perf_counter() - begin
        begin = time.perf_counter()
        scaler.step(optimizer)
        scaler.update()
        synchronize()
        optimizer_seconds = time.perf_counter() - begin
        return forward, backward, optimizer_seconds, float(loss.detach())

    step()  # Warmup is intentionally excluded.
    samples = [step() for _ in range(iterations)]
    torch.cuda.empty_cache()
    fields = tuple(zip(*samples, strict=True))
    names = ("forward", "backward", "optimizer", "loss")
    report: dict[str, float | str] = {"backend": backend}
    for name, values in zip(names, fields, strict=True):
        report[f"{name}_median"] = statistics.median(values)
        report[f"{name}_minimum"] = min(values)
        report[f"{name}_maximum"] = max(values)
    report["step_median"] = sum(report[f"{name}_median"] for name in names[:3])
    report["tokens_per_second"] = 512 / float(report["step_median"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    for backend in ("standard", "foreach", "fused"):
        try:
            print(measure_backend(backend, args.iterations))
        except (RuntimeError, ValueError) as error:
            print({"backend": backend, "status": "unsupported", "error": str(error)})


if __name__ == "__main__":
    main()
