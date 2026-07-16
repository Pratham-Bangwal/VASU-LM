"""Deterministic, in-memory planning utilities for weighted data mixtures."""

from __future__ import annotations

import random
from typing import Iterator

from .schemas import MixtureManifest, SamplingPlan, SourceAllocation
from .validation import calculate_token_allocations, validate_manifest


def build_sampling_plan(manifest: MixtureManifest) -> SamplingPlan:
    """Validate a manifest and build an exact, reproducible token allocation."""
    validate_manifest(manifest)
    token_allocations = calculate_token_allocations(manifest)
    allocations = tuple(
        SourceAllocation(
            source_id=source.id,
            domain=source.domain,
            weight=float(source.weight),
            allocated_tokens=token_allocations[source.id],
            available_tokens=source.token_count,
            replacement_required=(
                token_allocations[source.id] > source.token_count
            ),
        )
        for source in manifest.sources
    )
    source_order = [source.id for source in manifest.sources]
    random.Random(manifest.random_seed).shuffle(source_order)
    return SamplingPlan(
        manifest_version=manifest.manifest_version,
        experiment_id=manifest.experiment_id,
        random_seed=manifest.random_seed,
        target_tokens=manifest.target_tokens,
        sampling_with_replacement=manifest.sampling_with_replacement,
        allocations=allocations,
        deterministic_source_order=tuple(source_order),
    )


def iter_weighted_source_ids(
    manifest: MixtureManifest,
    draw_count: int,
) -> Iterator[str]:
    """Yield reproducible weighted source draws without touching source files."""
    if isinstance(draw_count, bool) or not isinstance(draw_count, int):
        raise TypeError("draw_count must be an integer")
    if draw_count < 0:
        raise ValueError("draw_count must be non-negative")
    validate_manifest(manifest)
    generator = random.Random(manifest.random_seed)
    source_ids = [source.id for source in manifest.sources]
    weights = [source.weight for source in manifest.sources]
    yield from generator.choices(source_ids, weights=weights, k=draw_count)
