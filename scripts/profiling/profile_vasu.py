"""Short, synthetic CUDA profile for VASU model throughput and memory.

This tool never loads or writes experiment checkpoints, datasets, or logs. It
uses random token IDs and ephemeral model/optimizer state so it is safe to run
alongside the repository's real training artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.cache import KVCache, PreallocatedKVCache  # noqa: E402
from vasu.config import ModelConfig, get_vasu_60m_config  # noqa: E402
from vasu.model.model import VASUModel  # noqa: E402


DEFAULT_BATCH_SIZE = 2
DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_PROMPT_LENGTH = 32
DEFAULT_NEW_TOKENS = 16


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA work before a wall-clock timing boundary."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_snapshot(device: torch.device) -> dict[str, float] | None:
    """Return allocated/reserved CUDA memory in MiB, when available."""

    if device.type != "cuda":
        return None
    mib = 1024**2
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / mib,
        "reserved_mib": torch.cuda.memory_reserved(device) / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / mib,
    }


def timed_training_step(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Measure one AMP forward/backward/optimizer step with synchronization."""

    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    started = time.perf_counter()
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(inputs)
        loss = functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten()
        )
    synchronize(device)
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scaler.scale(loss).backward()
    synchronize(device)
    backward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    synchronize(device)
    optimizer_seconds = time.perf_counter() - started
    return {
        "loss": float(loss.detach().cpu()),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "optimizer_seconds": optimizer_seconds,
        "step_seconds": forward_seconds + backward_seconds + optimizer_seconds,
    }


@torch.inference_mode()
def greedy_generation(
    model: VASUModel,
    prompt: torch.Tensor,
    new_tokens: int,
    *,
    cache_implementation: str,
    device: torch.device,
) -> dict[str, Any]:
    """Measure greedy generation and retain IDs for exact cache parity."""

    history = prompt
    first_token_seconds: float | None = None
    synchronize(device)
    started = time.perf_counter()
    if cache_implementation != "uncached":
        if cache_implementation == "dynamic":
            cache = KVCache(model.config.n_layers, model.config.max_seq_len)
        elif cache_implementation == "preallocated":
            cache = PreallocatedKVCache(
                model.config.n_layers,
                model.config.max_seq_len,
                prompt.size(0),
                model.config.n_heads,
                model.config.dim // model.config.n_heads,
                device=prompt.device,
                dtype=next(model.parameters()).dtype,
            )
        else:
            raise ValueError(f"Unsupported cache implementation: {cache_implementation}")
        logits = model(history, kv_cache=cache, cache_mode="prefill")
        synchronize(device)
        first_token_seconds = time.perf_counter() - started
        for _ in range(new_tokens):
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            history = torch.cat((history, token), dim=1)
            if history.size(1) >= model.config.max_seq_len:
                break
            logits = model(token, kv_cache=cache, cache_mode="decode")
    else:
        for index in range(new_tokens):
            logits = model(history)
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            history = torch.cat((history, token), dim=1)
            if index == 0:
                synchronize(device)
                first_token_seconds = time.perf_counter() - started
            if history.size(1) >= model.config.max_seq_len:
                break
    synchronize(device)
    elapsed = time.perf_counter() - started
    generated = history[:, prompt.size(1) :]
    count = generated.size(1)
    return {
        "token_ids": generated.cpu().tolist(),
        "generated_tokens": count,
        "first_token_seconds": first_token_seconds,
        "total_seconds": elapsed,
        "tokens_per_second": count / elapsed if elapsed else 0.0,
    }


def profile_model(
    name: str,
    config: ModelConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    new_tokens: int,
) -> dict[str, Any]:
    """Profile one bounded synthetic training and inference workload."""

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = VASUModel(config).to(device)
    synchronize(device)
    load_seconds = time.perf_counter() - started
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    inputs = torch.randint(
        config.vocab_size, (batch_size, sequence_length), device=device
    )
    targets = torch.randint_like(inputs, high=config.vocab_size)

    model.train()
    _ = timed_training_step(model, optimizer, scaler, inputs, targets, device)
    training = timed_training_step(
        model, optimizer, scaler, inputs, targets, device
    )
    training["tokens_per_second"] = (
        batch_size * sequence_length / training["step_seconds"]
    )
    training["memory"] = memory_snapshot(device)

    prompt = inputs[:1, : min(DEFAULT_PROMPT_LENGTH, sequence_length)]
    _ = greedy_generation(model, prompt, 2, cache_implementation="uncached", device=device)
    _ = greedy_generation(model, prompt, 2, cache_implementation="dynamic", device=device)
    _ = greedy_generation(model, prompt, 2, cache_implementation="preallocated", device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    uncached = greedy_generation(
        model, prompt, new_tokens, cache_implementation="uncached", device=device
    )
    uncached["memory"] = memory_snapshot(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dynamic = greedy_generation(
        model, prompt, new_tokens, cache_implementation="dynamic", device=device
    )
    dynamic["memory"] = memory_snapshot(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    preallocated = greedy_generation(
        model, prompt, new_tokens, cache_implementation="preallocated", device=device
    )
    preallocated["memory"] = memory_snapshot(device)
    uncached_ids = uncached.pop("token_ids")
    dynamic["exact_greedy_parity"] = dynamic.pop("token_ids") == uncached_ids
    preallocated["exact_greedy_parity"] = (
        preallocated.pop("token_ids") == uncached_ids
    )

    result = {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_construction_seconds": load_seconds,
        "training": training,
        "inference_uncached": uncached,
        "inference_dynamic_cache": dynamic,
        "inference_preallocated_cache": preallocated,
    }
    del scaler, optimizer, inputs, targets, prompt, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.new_tokens) < 1:
        parser.error("batch size, sequence length, and new tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    report = {
        "device": str(device),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "sdpa": {
            "flash_enabled": (
                torch.backends.cuda.flash_sdp_enabled()
                if device.type == "cuda"
                else None
            ),
            "memory_efficient_enabled": (
                torch.backends.cuda.mem_efficient_sdp_enabled()
                if device.type == "cuda"
                else None
            ),
            "math_enabled": (
                torch.backends.cuda.math_sdp_enabled()
                if device.type == "cuda"
                else None
            ),
        },
        "models": [
            profile_model(
                "VASU-31M", ModelConfig(), device, args.batch_size,
                args.sequence_length, args.new_tokens,
            ),
            profile_model(
                "VASU-60M", get_vasu_60m_config(), device, args.batch_size,
                args.sequence_length, args.new_tokens,
            ),
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
