"""Bounded, no-update CPU runtime qualification for VASU-140M-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.cache import KVCache, PreallocatedKVCache  # noqa: E402
from vasu.config import get_vasu_140m_config  # noqa: E402
from vasu.model import VASUModel, validate_family_config  # noqa: E402

SEED = 140_042
BATCH_SIZE = 1
SEQUENCE_LENGTH = 8
PROMPT_TOKENS = (101, 202, 303, 404)
DECODE_TOKENS = (505, 606, 707)
RTOL = 1e-4
ATOL = 1e-5


def run_forward_backward(
    model: VASUModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    """Run one synthetic forward/backward pass without an optimizer."""
    model.train()
    model.zero_grad(set_to_none=True)

    started = time.perf_counter()
    logits = model(inputs)
    forward_seconds = time.perf_counter() - started
    loss = functional.cross_entropy(
        logits.flatten(0, 1),
        targets.flatten(),
    )

    started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter() - started

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    parameter_tensor_count = sum(1 for _ in model.parameters())
    finite_gradients = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    report = {
        "input_shape": list(inputs.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "finite_gradients": finite_gradients,
        "gradient_tensor_count": len(gradients),
        "parameter_tensor_count": parameter_tensor_count,
        "all_parameters_received_gradients": (
            len(gradients) == parameter_tensor_count
        ),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_created": False,
        "optimizer_step_performed": False,
    }
    model.zero_grad(set_to_none=True)
    report["gradients_cleared"] = all(
        parameter.grad is None for parameter in model.parameters()
    )
    return report


def _preallocated_cache(model: VASUModel) -> PreallocatedKVCache:
    config = model.config
    parameter = next(model.parameters())
    return PreallocatedKVCache(
        config.n_layers,
        config.max_seq_len,
        batch_size=1,
        n_heads=config.n_heads,
        head_dim=config.dim // config.n_heads,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def _max_absolute_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    return float((actual - expected).abs().max())


@torch.inference_mode()
def run_cache_parity(
    model: VASUModel,
    prompt: torch.Tensor,
    decode_tokens: tuple[int, ...],
) -> dict[str, Any]:
    """Compare both cache implementations with full-sequence execution."""
    model.eval()
    dynamic = KVCache(model.config.n_layers, model.config.max_seq_len)
    preallocated = _preallocated_cache(model)

    uncached_prefill = model(prompt)
    dynamic_prefill = model(
        prompt,
        kv_cache=dynamic,
        cache_mode="prefill",
    )
    preallocated_prefill = model(
        prompt,
        kv_cache=preallocated,
        cache_mode="prefill",
    )
    torch.testing.assert_close(
        dynamic_prefill,
        uncached_prefill,
        rtol=RTOL,
        atol=ATOL,
    )
    torch.testing.assert_close(
        preallocated_prefill,
        uncached_prefill,
        rtol=RTOL,
        atol=ATOL,
    )

    dynamic_errors = [
        _max_absolute_error(dynamic_prefill, uncached_prefill)
    ]
    preallocated_errors = [
        _max_absolute_error(preallocated_prefill, uncached_prefill)
    ]
    history = prompt
    for token_id in decode_tokens:
        newest = torch.tensor(
            [[token_id]],
            device=prompt.device,
            dtype=prompt.dtype,
        )
        history = torch.cat((history, newest), dim=1)
        uncached = model(history)[:, -1, :]
        dynamic_logits = model(
            newest,
            kv_cache=dynamic,
            cache_mode="decode",
        )[:, -1, :]
        preallocated_logits = model(
            newest,
            kv_cache=preallocated,
            cache_mode="decode",
        )[:, -1, :]
        torch.testing.assert_close(
            dynamic_logits,
            uncached,
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            preallocated_logits,
            uncached,
            rtol=RTOL,
            atol=ATOL,
        )
        dynamic_errors.append(
            _max_absolute_error(dynamic_logits, uncached)
        )
        preallocated_errors.append(
            _max_absolute_error(preallocated_logits, uncached)
        )

    return {
        "prompt_length": prompt.size(1),
        "decode_steps": len(decode_tokens),
        "final_sequence_length": history.size(1),
        "rtol": RTOL,
        "atol": ATOL,
        "dynamic_cache": {
            "passed": True,
            "maximum_absolute_error": max(dynamic_errors),
            "final_cache_length": dynamic.sequence_length,
        },
        "preallocated_cache": {
            "passed": True,
            "maximum_absolute_error": max(preallocated_errors),
            "final_cache_length": preallocated.sequence_length,
            "allocation_bytes": preallocated.allocation_bytes,
        },
    }


def build_qualification_report() -> dict[str, Any]:
    """Execute the bounded VASU-140M-v1 CPU qualification."""
    torch.manual_seed(SEED)
    config = get_vasu_140m_config()
    family = validate_family_config("vasu_140m_v1", config)

    started = time.perf_counter()
    model = VASUModel(config).to(device="cpu", dtype=torch.float32)
    construction_seconds = time.perf_counter() - started
    state_keys_before = tuple(model.state_dict())

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    inputs = torch.randint(
        config.vocab_size,
        (BATCH_SIZE, SEQUENCE_LENGTH),
        generator=generator,
    )
    targets = torch.randint(
        config.vocab_size,
        (BATCH_SIZE, SEQUENCE_LENGTH),
        generator=generator,
    )
    forward_backward = run_forward_backward(model, inputs, targets)
    cache_parity = run_cache_parity(
        model,
        torch.tensor([PROMPT_TOKENS], dtype=torch.long),
        DECODE_TOKENS,
    )
    state_keys_after = tuple(model.state_dict())

    checks = {
        "family_identity": family.family_id == "vasu_140m_v1",
        "parameter_count": (
            sum(parameter.numel() for parameter in model.parameters())
            == family.expected_parameter_count
        ),
        "finite_forward_backward": all(
            forward_backward[name]
            for name in (
                "finite_logits",
                "finite_loss",
                "finite_gradients",
                "all_parameters_received_gradients",
                "gradients_cleared",
            )
        ),
        "dynamic_cache_parity": cache_parity["dynamic_cache"]["passed"],
        "preallocated_cache_parity": (
            cache_parity["preallocated_cache"]["passed"]
        ),
        "model_state_keys_unchanged": state_keys_before == state_keys_after,
        "no_optimizer_update": (
            not forward_backward["optimizer_created"]
            and not forward_backward["optimizer_step_performed"]
        ),
    }
    return {
        "schema": "vasu.model-family-cpu-qualification.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "family_id": family.family_id,
        "family_fingerprint": family.family_fingerprint,
        "config_fingerprint": family.config_fingerprint,
        "parameter_count": family.expected_parameter_count,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float32",
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
        },
        "workload": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "prompt_tokens": list(PROMPT_TOKENS),
            "decode_tokens": list(DECODE_TOKENS),
        },
        "construction_seconds": construction_seconds,
        "forward_backward": forward_backward,
        "cache_parity": cache_parity,
        "checks": checks,
        "passed": all(checks.values()),
        "training_authorized": False,
        "remaining_gates": [
            "cuda_memory_throughput_thermal",
            "checkpoint_round_trip_wrong_family_rejection",
            "exact_resume",
            "513_token_data_release",
            "frozen_evaluation_baselines",
            "scientific_plan_and_hash_bound_authorization",
        ],
    }


def report_sha256(report: dict[str, Any]) -> str:
    """Hash the canonical payload, excluding its self-referential hash."""
    payload = {
        key: value
        for key, value in report.items()
        if key != "report_sha256"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_immutable_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically publish a new report without overwriting existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite CPU qualification report: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new immutable JSON evidence path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_qualification_report()
    report["report_sha256"] = report_sha256(report)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output is not None:
        write_immutable_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
