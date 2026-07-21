"""Construction, forward-pass, and optional CUDA training smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


# Allow this file to be run directly from the repository root on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.config import ModelConfig, get_vasu_60m_config
from vasu.model.model import VASUModel


EXPECTED_VASU_31M_PARAMETERS = 31_168_896
EXPECTED_VASU_60M_PARAMETERS = 58_337_792
COMPATIBILITY_BATCH_SIZE = 1
COMPATIBILITY_SEQUENCE_LENGTH = 16
REALISTIC_SEQUENCE_LENGTH = 256
PRIMARY_BATCH_SIZE = 2
FALLBACK_BATCH_SIZE = 1


def count_parameters(model: torch.nn.Module) -> int:
    """Count unique trainable parameters, including tied weights only once."""
    return sum(parameter.numel() for parameter in model.parameters())


def run_forward_smoke_test(
    name: str,
    config: ModelConfig,
    expected_parameters: int,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
) -> int:
    model = VASUModel(config).to(device)
    parameter_count = count_parameters(model)

    if parameter_count != expected_parameters:
        raise AssertionError(
            f"{name} parameter count changed: expected "
            f"{expected_parameters:,}, got {parameter_count:,}."
        )

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        device=device,
    )

    model.eval()
    with torch.inference_mode():
        logits = model(input_ids)

    expected_shape = (
        batch_size,
        sequence_length,
        config.vocab_size,
    )
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"{name} produced {tuple(logits.shape)}, expected "
            f"{expected_shape}."
        )

    print(f"{name} parameters: {parameter_count:,}")
    print(
        f"{name} forward pass: OK "
        f"(batch_size={batch_size}, seq_len={sequence_length}, "
        f"shape={tuple(logits.shape)})"
    )
    if device.type == "cuda":
        print_cuda_memory(f"{name} forward pass")

    del logits, input_ids, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return parameter_count


def mib(byte_count: int) -> float:
    return byte_count / (1024**2)


def print_cuda_memory(label: str) -> None:
    print(
        f"{label} CUDA memory: "
        f"allocated={mib(torch.cuda.memory_allocated()):.2f} MiB, "
        f"reserved={mib(torch.cuda.memory_reserved()):.2f} MiB, "
        f"peak={mib(torch.cuda.max_memory_allocated()):.2f} MiB"
    )


def run_cuda_training_step(
    config: ModelConfig,
    batch_size: int,
    sequence_length: int,
) -> None:
    """Run one realistic AMP optimizer step without checkpoint I/O."""
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = VASUModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        device=device,
    )
    targets = torch.randint(
        0,
        config.vocab_size,
        (batch_size, sequence_length),
        device=device,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast("cuda", enabled=True):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            targets.reshape(-1),
        )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if not torch.isfinite(loss):
        raise AssertionError(f"CUDA training loss is not finite: {loss.item()}")

    print(
        "VASU-60M CUDA training step: OK "
        f"(batch_size={batch_size}, seq_len={sequence_length}, "
        f"loss={loss.item():.6f})"
    )
    print_cuda_memory("VASU-60M training step")


def clear_cuda_after_oom() -> None:
    """Release cached CUDA blocks after a failed smoke-test attempt."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    default_config = ModelConfig()
    vasu_60m_config = get_vasu_60m_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Smoke-test device: {device}")

    run_forward_smoke_test(
        "VASU-31M",
        default_config,
        EXPECTED_VASU_31M_PARAMETERS,
        device,
        COMPATIBILITY_BATCH_SIZE,
        COMPATIBILITY_SEQUENCE_LENGTH,
    )

    if device.type == "cuda":
        batch_2_forward_feasible = False
        try:
            run_forward_smoke_test(
                "VASU-60M",
                vasu_60m_config,
                EXPECTED_VASU_60M_PARAMETERS,
                device,
                PRIMARY_BATCH_SIZE,
                REALISTIC_SEQUENCE_LENGTH,
            )
            batch_2_forward_feasible = True
        except torch.OutOfMemoryError:
            print(
                "VASU-60M forward pass: CUDA OOM "
                "(batch_size=2, seq_len=256)"
            )
            clear_cuda_after_oom()

        batch_2_feasible = False
        if batch_2_forward_feasible:
            try:
                run_cuda_training_step(
                    vasu_60m_config,
                    PRIMARY_BATCH_SIZE,
                    REALISTIC_SEQUENCE_LENGTH,
                )
                batch_2_feasible = True
            except torch.OutOfMemoryError:
                print(
                    "VASU-60M CUDA training step: OOM "
                    "(batch_size=2, seq_len=256)"
                )
                clear_cuda_after_oom()

        print(f"batch_size=2 feasible: {batch_2_feasible}")

        if batch_2_feasible:
            print("fallback batch_size=1 feasible: not tested (not needed)")
        else:
            fallback_feasible = False
            try:
                run_cuda_training_step(
                    vasu_60m_config,
                    FALLBACK_BATCH_SIZE,
                    REALISTIC_SEQUENCE_LENGTH,
                )
                fallback_feasible = True
            except torch.OutOfMemoryError:
                print(
                    "VASU-60M CUDA training step: OOM "
                    "(batch_size=1, seq_len=256)"
                )
                clear_cuda_after_oom()

            print(f"fallback batch_size=1 feasible: {fallback_feasible}")
    else:
        run_forward_smoke_test(
            "VASU-60M",
            vasu_60m_config,
            EXPECTED_VASU_60M_PARAMETERS,
            device,
            PRIMARY_BATCH_SIZE,
            REALISTIC_SEQUENCE_LENGTH,
        )
        print("VASU-60M CUDA training step: SKIPPED (CUDA unavailable)")
        print("CUDA memory: unavailable")
        print("batch_size=2 feasible: not tested on CUDA")
        print("fallback batch_size=1 feasible: not tested on CUDA")

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
