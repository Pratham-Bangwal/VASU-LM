"""Bounded, no-update CUDA/AMP qualification for VASU-140M-v1.

This script is deliberately synthetic: it never reads a dataset, constructs an
optimizer, or writes beneath the repository checkpoint tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qualify_vasu_140m_cpu import report_sha256, write_immutable_report  # noqa: E402
from vasu.config import get_vasu_140m_config  # noqa: E402
from vasu.model import (  # noqa: E402
    VASUModel,
    build_model_family_identity,
    load_family_model_state,
    validate_checkpoint_family_identity,
    validate_family_config,
)
from vasu.training.checkpoint import save_checkpoint  # noqa: E402

SEED = 140_044
BATCH_SIZE = 1
SEQUENCE_LENGTH = 512
WARMUP_ITERATIONS = 3
MEASURED_ITERATIONS = 5
MINIMUM_FREE_DISK_BYTES = 2 * 1024**3
THERMAL_STOP_CELSIUS = 88.0


def select_amp_dtype(device: int) -> tuple[torch.dtype, str | None]:
    """Prefer BF16, recording the explicit FP16 fallback."""
    if torch.cuda.is_bf16_supported(device):
        return torch.bfloat16, None
    return torch.float16, "cuda_bf16_not_supported"


def require_cuda(device: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("VASU-140M CUDA qualification requires available CUDA")
    if device < 0 or device >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is invalid: {device}")
    return torch.device(f"cuda:{device}")


def synthetic_tokens(config_vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    shape = (BATCH_SIZE, SEQUENCE_LENGTH)
    inputs = torch.randint(config_vocab_size, shape, generator=generator)
    targets = torch.randint(config_vocab_size, shape, generator=generator)
    return inputs.to(device), targets.to(device)


def read_telemetry(device: int) -> dict[str, Any]:
    """Read one optional NVIDIA telemetry sample without making it a fallback."""
    command = ["nvidia-smi", f"--id={device}", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
        fields = [field.strip() for field in completed.stdout.strip().split(",")]
        return {"available": True, "temperature_celsius": float(fields[0]), "power_watts": float(fields[1]), "provider": "nvidia-smi"}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        return {"available": False, "provider": "nvidia-smi", "error": str(error)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_iteration(model: VASUModel, inputs: torch.Tensor, targets: torch.Tensor, dtype: torch.dtype) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(inputs.device)
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(inputs)
        loss = functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    torch.cuda.synchronize(inputs.device)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(inputs.device)
    backward_seconds = time.perf_counter() - started
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite = bool(torch.isfinite(logits).all() and torch.isfinite(loss)) and bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    maximum_gradient_norm = max((float(gradient.norm().detach().cpu()) for gradient in gradients), default=0.0)
    model.zero_grad(set_to_none=True)
    return {"forward_seconds": forward_seconds, "backward_seconds": backward_seconds, "finite": finite, "maximum_gradient_norm": maximum_gradient_norm, "gradients_cleared": all(parameter.grad is None for parameter in model.parameters())}


def build_qualification_report(device_index: int = 0) -> dict[str, Any]:
    """Execute the reviewed bounded CUDA workload. Do not use for training."""
    device = require_cuda(device_index)
    config = get_vasu_140m_config()
    family = validate_family_config("vasu_140m_v1", config)
    free_before = shutil.disk_usage(tempfile.gettempdir()).free
    if free_before < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("CUDA qualification requires at least 2 GiB free temporary disk")
    dtype, fallback = select_amp_dtype(device_index)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    telemetry = [read_telemetry(device_index)]
    if telemetry[0].get("temperature_celsius", 0.0) >= THERMAL_STOP_CELSIUS:
        raise RuntimeError("thermal threshold reached before CUDA qualification")
    torch.cuda.reset_peak_memory_stats(device)
    model = VASUModel(config).to(device=device, dtype=torch.float32)
    state_keys = tuple(model.state_dict())
    inputs, targets = synthetic_tokens(config.vocab_size, device)
    for _ in range(WARMUP_ITERATIONS):
        _run_iteration(model, inputs, targets, dtype)
    observations: list[dict[str, Any]] = []
    for _ in range(MEASURED_ITERATIONS):
        sample = _run_iteration(model, inputs, targets, dtype)
        sample["telemetry"] = read_telemetry(device_index)
        telemetry.append(sample["telemetry"])
        if sample["telemetry"].get("temperature_celsius", 0.0) >= THERMAL_STOP_CELSIUS:
            raise RuntimeError("thermal threshold reached during CUDA qualification")
        observations.append(sample)
    torch.cuda.synchronize(device)
    temporary = Path(tempfile.mkdtemp(prefix="vasu_140m_cuda_amp_"))
    checkpoint_path = temporary / "model_only.pt"
    try:
        model = model.to("cpu")
        started = time.perf_counter()
        save_checkpoint(model, None, None, 0, None, checkpoint_path, global_step=0, verify_after_write=True, fsync=True, model_family_identity=build_model_family_identity(family.family_id, config), training_authorized=False, qualification_only=True)
        write_seconds = time.perf_counter() - started
        started = time.perf_counter()
        payload = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        validate_checkpoint_family_identity(payload, family.family_id)
        restored = VASUModel(config)
        load_family_model_state(restored, payload, family.family_id)
        reload_seconds = time.perf_counter() - started
        checkpoint = {"bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path), "write_seconds": write_seconds, "reload_seconds": reload_seconds, "strict_reload": True}
    except Exception:
        raise RuntimeError(f"qualification temporary directory preserved: {temporary}") from None
    shutil.rmtree(temporary)
    checks = {"family_identity": True, "finite_iterations": all(item["finite"] for item in observations), "gradients_cleared": all(item["gradients_cleared"] for item in observations), "state_keys_unchanged": state_keys == tuple(model.state_dict()), "checkpoint_cleanup": not temporary.exists(), "strict_checkpoint_reload": checkpoint["strict_reload"], "no_optimizer_update": True}
    elapsed = sum(item["forward_seconds"] + item["backward_seconds"] for item in observations)
    return {"schema": "vasu.model-family-cuda-amp-qualification.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "family_id": family.family_id, "family_fingerprint": family.family_fingerprint, "config_fingerprint": family.config_fingerprint, "parameter_count": family.expected_parameter_count, "environment": {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda_device": device_index, "cuda_name": torch.cuda.get_device_name(device), "amp_dtype": str(dtype).replace("torch.", ""), "amp_fallback_reason": fallback}, "workload": {"seed": SEED, "synthetic": True, "batch_size": BATCH_SIZE, "sequence_length": SEQUENCE_LENGTH, "warmup_iterations": WARMUP_ITERATIONS, "measured_iterations": MEASURED_ITERATIONS, "optimizer_created": False, "optimizer_updates": 0}, "metrics": {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "tokens_per_second": (BATCH_SIZE * SEQUENCE_LENGTH * MEASURED_ITERATIONS) / elapsed, "observations": observations}, "telemetry": telemetry, "checkpoint": checkpoint, "free_disk_bytes_before": free_before, "free_disk_bytes_after": shutil.disk_usage(tempfile.gettempdir()).free, "checks": checks, "passed": all(checks.values()), "training_authorized": False, "remaining_gates": ["base_pretraining_data_release", "frozen_base_model_evaluations", "real_data_exact_resume", "scientific_plan_and_hash_bound_authorization"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_qualification_report(args.device)
    report["report_sha256"] = report_sha256(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.output:
        write_immutable_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
