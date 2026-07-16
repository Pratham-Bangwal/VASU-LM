"""Benchmark greedy cached and uncached generation on the preferred model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from vasu.cache import KVCache
from vasu.config import get_vasu_60m_config
from vasu.data.prompt_templates import format_prompt
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer


CHECKPOINT = Path(
    "checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt"
)
TOKENIZER = Path("assets/tokenizer.json")
OUTPUT = Path("evaluation/kv_cache_benchmark.json")
PROMPTS = (
    "What is the internet? Explain it in simple words.",
    "Give exactly 3 benefits of education.",
    "Write a short story about a robot.",
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def greedy_uncached(
    model: VASUModel,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
) -> tuple[list[int], float]:
    history = prompt_ids
    synchronize(prompt_ids.device)
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        if history.size(1) >= model.config.max_seq_len:
            break
        logits = model(history)
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        history = torch.cat((history, token), dim=1)
        if token.item() == eos_id:
            break
    synchronize(prompt_ids.device)
    elapsed = time.perf_counter() - start
    return history[0, prompt_ids.size(1):].tolist(), elapsed


def greedy_cached(
    model: VASUModel,
    prompt_ids: torch.Tensor,
    eos_id: int,
    max_new_tokens: int,
) -> tuple[list[int], float, float, float]:
    cache = KVCache(model.config.n_layers, model.config.max_seq_len)
    history = prompt_ids
    synchronize(prompt_ids.device)
    total_start = time.perf_counter()
    prefill_start = total_start
    logits = model(history, kv_cache=cache, cache_mode="prefill")
    synchronize(prompt_ids.device)
    prefill_time = time.perf_counter() - prefill_start
    decode_time = 0.0
    for _ in range(max_new_tokens):
        if history.size(1) >= model.config.max_seq_len:
            break
        token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        history = torch.cat((history, token), dim=1)
        if token.item() == eos_id or history.size(1) >= model.config.max_seq_len:
            break
        synchronize(prompt_ids.device)
        decode_start = time.perf_counter()
        logits = model(token, kv_cache=cache, cache_mode="decode")
        synchronize(prompt_ids.device)
        decode_time += time.perf_counter() - decode_start
    synchronize(prompt_ids.device)
    total_time = time.perf_counter() - total_start
    return (
        history[0, prompt_ids.size(1):].tolist(),
        prefill_time,
        decode_time,
        total_time,
    )


def peak_memory_mib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER))
    model = VASUModel(get_vasu_60m_config()).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    global_step = int(checkpoint["global_step"])
    del checkpoint
    model.eval()
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    if eos_id is None:
        raise RuntimeError("tokenizer has no EOS token")

    results = []
    with torch.inference_mode():
        # Warm both SDPA shapes before timing so the first measured mode does
        # not pay one-time CUDA/kernel initialization costs alone.
        warm_prompt = format_prompt(PROMPTS[0], prompt_format="alpaca")
        warm_ids = torch.tensor(
            [tokenizer.encode(warm_prompt)], device=device
        )
        _ = model(warm_ids)
        warm_cache = KVCache(model.config.n_layers, model.config.max_seq_len)
        warm_logits = model(
            warm_ids, kv_cache=warm_cache, cache_mode="prefill"
        )
        warm_token = torch.argmax(
            warm_logits[:, -1, :], dim=-1, keepdim=True
        )
        _ = model(warm_token, kv_cache=warm_cache, cache_mode="decode")
        synchronize(device)
        del warm_cache, warm_logits, warm_token, warm_ids

        for prompt in PROMPTS:
            formatted = format_prompt(prompt, prompt_format="alpaca")
            prompt_ids = torch.tensor(
                [tokenizer.encode(formatted)], device=device
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            uncached_ids, uncached_total = greedy_uncached(
                model, prompt_ids, eos_id, args.max_new_tokens
            )
            uncached_peak = peak_memory_mib(device)

            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            cached_ids, prefill, decode, cached_total = greedy_cached(
                model, prompt_ids, eos_id, args.max_new_tokens
            )
            cached_peak = peak_memory_mib(device)
            parity = cached_ids == uncached_ids
            if not parity:
                raise RuntimeError(f"greedy token parity failed for: {prompt}")
            generated_count = len(cached_ids)
            results.append(
                {
                    "prompt": prompt,
                    "prompt_tokens": int(prompt_ids.size(1)),
                    "generated_tokens": generated_count,
                    "generated_token_ids": cached_ids,
                    "parity": parity,
                    "uncached": {
                        "total_seconds": uncached_total,
                        "tokens_per_second": (
                            generated_count / uncached_total
                            if uncached_total else None
                        ),
                        "peak_cuda_memory_mib": uncached_peak,
                    },
                    "cached": {
                        "prefill_seconds": prefill,
                        "decode_seconds": decode,
                        "total_seconds": cached_total,
                        "tokens_per_second": (
                            generated_count / cached_total
                            if cached_total else None
                        ),
                        "peak_cuda_memory_mib": cached_peak,
                    },
                }
            )

    report = {
        "checkpoint": str(CHECKPOINT),
        "global_step": global_step,
        "device": str(device),
        "decoding": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "all_token_ids_match": all(item["parity"] for item in results),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark: {OUTPUT}")


if __name__ == "__main__":
    main()
