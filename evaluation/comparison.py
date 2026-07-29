"""Deterministic paired evaluation statistics for experiment reports."""

from __future__ import annotations

import random
from typing import Sequence


def paired_binary_bootstrap(
    baseline_correct: Sequence[bool],
    candidate_correct: Sequence[bool],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Return a reproducible percentile interval for paired accuracy delta."""

    if len(baseline_correct) != len(candidate_correct) or not baseline_correct:
        raise ValueError("paired non-empty outcome sequences are required")
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    differences = [
        int(candidate) - int(baseline)
        for baseline, candidate in zip(baseline_correct, candidate_correct, strict=True)
    ]
    rng = random.Random(seed)
    size = len(differences)
    draws = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    return {
        "examples": size,
        "baseline_accuracy": sum(baseline_correct) / size,
        "candidate_accuracy": sum(candidate_correct) / size,
        "delta": sum(differences) / size,
        "ci_95_lower": draws[int(0.025 * samples)],
        "ci_95_upper": draws[int(0.975 * samples) - 1],
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }
