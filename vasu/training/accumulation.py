"""Helpers for correct, count-aware gradient accumulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

import torch


@dataclass(frozen=True)
class AccumulationPlan:
    """Expected optimizer-step and token accounting for a finite dataset."""

    records: int
    microbatches: int
    optimizer_steps: int
    final_microbatches: int
    final_records: int
    trained_tokens: int
    final_tokens: int


def build_accumulation_plan(
    *,
    record_count: int,
    batch_size: int,
    accumulation_steps: int,
    sequence_length: int,
) -> AccumulationPlan:
    """Describe a non-dropping DataLoader run, including its partial tail."""

    for name, value in (
        ("record_count", record_count),
        ("batch_size", batch_size),
        ("accumulation_steps", accumulation_steps),
        ("sequence_length", sequence_length),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    microbatches = math.ceil(record_count / batch_size)
    optimizer_steps = math.ceil(microbatches / accumulation_steps)
    remainder = microbatches % accumulation_steps
    final_microbatches = remainder or accumulation_steps
    records_before_final = (
        (optimizer_steps - 1) * accumulation_steps * batch_size
    )
    final_records = record_count - records_before_final
    return AccumulationPlan(
        records=record_count,
        microbatches=microbatches,
        optimizer_steps=optimizer_steps,
        final_microbatches=final_microbatches,
        final_records=final_records,
        trained_tokens=record_count * sequence_length,
        final_tokens=final_records * sequence_length,
    )


def normalize_partial_accumulation(
    parameters: Iterable[torch.nn.Parameter],
    *,
    accumulated_microbatches: int,
    target_microbatches: int,
) -> float:
    """Correct gradients after a partial flush and AMP unscaling.

    Every microbatch loss is divided by ``target_microbatches`` before
    backward. A partial tail therefore needs a ``target / actual`` correction
    before clipping and the optimizer update. Full accumulations are unchanged.
    """

    if accumulated_microbatches < 1:
        raise ValueError("accumulated_microbatches must be positive")
    if target_microbatches < 1:
        raise ValueError("target_microbatches must be positive")
    if accumulated_microbatches > target_microbatches:
        raise ValueError("accumulated_microbatches cannot exceed target_microbatches")

    factor = target_microbatches / accumulated_microbatches
    if factor != 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
    return factor
