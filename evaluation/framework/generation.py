"""Deterministic and seeded generation execution for capability suites."""

from __future__ import annotations

import random
import time
from typing import Any

import torch

from vasu.inference.generate import generate


def generation_settings(mode: str, max_new_tokens: int) -> dict[str, Any]:
    if mode == "greedy":
        return {"mode": mode, "do_sample": False, "max_new_tokens": max_new_tokens}
    if mode == "sampled":
        return {"mode": mode, "do_sample": True, "max_new_tokens": max_new_tokens, "temperature": 0.45, "top_k": 20, "top_p": 0.8}
    raise ValueError("mode must be 'greedy' or 'sampled'.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def run_generation(*, model: Any, tokenizer: Any, prompt: str, prompt_format: str, device: torch.device, settings: dict[str, Any], seed: int | None) -> tuple[str, float]:
    if seed is not None:
        set_seed(seed)
    started = time.perf_counter()
    response = generate(model=model, tokenizer=tokenizer, prompt=prompt, device=device, prompt_format=prompt_format, **{key: value for key, value in settings.items() if key != "mode"})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return response, time.perf_counter() - started
