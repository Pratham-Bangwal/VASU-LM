"""Strict, actionable validation for VASU mixture manifests."""

from __future__ import annotations

from datetime import datetime
import math
import os
import re
from urllib.parse import urlparse

from .schemas import (
    MixtureManifest,
    SUPPORTED_DOMAINS,
    SUPPORTED_FORMATS,
    SUPPORTED_MANIFEST_VERSIONS,
)


WEIGHT_TOLERANCE = 1e-6
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class MixtureValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__(
            "Mixture manifest validation failed:\n- " + "\n- ".join(errors)
        )


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def calculate_token_allocations(manifest: MixtureManifest) -> dict[str, int]:
    """Allocate exact integer tokens with deterministic largest remainders.

    Validation rejects material weight-sum errors. Dividing by the accepted
    floating-point total here only absorbs representational noise and ensures
    integer allocations always add up exactly to ``target_tokens``.
    """
    weight_total = sum(float(source.weight) for source in manifest.sources)
    if not math.isfinite(weight_total) or weight_total <= 0:
        raise ValueError("source weights must have a finite, positive total")
    raw = {
        source.id: manifest.target_tokens * float(source.weight) / weight_total
        for source in manifest.sources
    }
    allocations = {source_id: math.floor(value) for source_id, value in raw.items()}
    remaining = manifest.target_tokens - sum(allocations.values())
    ranked = sorted(
        manifest.sources,
        key=lambda source: (-(raw[source.id] - allocations[source.id]), source.id),
    )
    for source in ranked[:remaining]:
        allocations[source.id] += 1
    return allocations


def validate_manifest(
    manifest: MixtureManifest,
    *,
    weight_tolerance: float = WEIGHT_TOLERANCE,
) -> None:
    errors: list[str] = []

    if not isinstance(manifest.manifest_version, str) or (
        manifest.manifest_version not in SUPPORTED_MANIFEST_VERSIONS
    ):
        errors.append(
            f"manifest_version {manifest.manifest_version!r} is unsupported; "
            f"supported versions: {sorted(SUPPORTED_MANIFEST_VERSIONS)}"
        )
    for name in ("experiment_id", "description", "tokenizer_path", "created_at"):
        if not _is_non_empty_string(getattr(manifest, name)):
            errors.append(f"manifest.{name} must be a non-empty string")
    if not isinstance(manifest.notes, str):
        errors.append("manifest.notes must be a string")
    if isinstance(manifest.target_tokens, bool) or not isinstance(
        manifest.target_tokens, int
    ) or manifest.target_tokens < 1:
        errors.append("manifest.target_tokens must be an integer of at least 1")
    if isinstance(manifest.random_seed, bool) or not isinstance(
        manifest.random_seed, int
    ) or not 0 <= manifest.random_seed <= (2**63 - 1):
        errors.append("manifest.random_seed must be an integer from 0 to 2^63-1")
    if not isinstance(manifest.sampling_with_replacement, bool):
        errors.append("manifest.sampling_with_replacement must be boolean")
    if not isinstance(manifest.allow_duplicate_paths, bool):
        errors.append("manifest.allow_duplicate_paths must be boolean")
    try:
        created_at = datetime.fromisoformat(manifest.created_at)
        if created_at.tzinfo is None:
            errors.append("manifest.created_at must include a timezone offset")
    except (TypeError, ValueError):
        errors.append("manifest.created_at must be a valid ISO-8601 timestamp")

    if not manifest.sources:
        errors.append("manifest.sources must contain at least one source")

    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, str] = {}
    total_weight = 0.0
    for index, source in enumerate(manifest.sources):
        prefix = f"sources[{index}] ({source.id!r})"
        if not _is_non_empty_string(source.id):
            errors.append(f"{prefix}.id must be a non-empty string")
        elif source.id in seen_ids:
            errors.append(
                f"duplicate source id {source.id!r} at indexes "
                f"{seen_ids[source.id]} and {index}"
            )
        else:
            seen_ids[source.id] = index

        if not isinstance(source.domain, str) or (
            source.domain not in SUPPORTED_DOMAINS
        ):
            errors.append(
                f"{prefix}.domain {source.domain!r} is unsupported; "
                f"choose one of {sorted(SUPPORTED_DOMAINS)}"
            )
        if not isinstance(source.format, str) or (
            source.format not in SUPPORTED_FORMATS
        ):
            errors.append(
                f"{prefix}.format {source.format!r} is unsupported; "
                f"choose one of {sorted(SUPPORTED_FORMATS)}"
            )
        if not _is_non_empty_string(source.path):
            errors.append(f"{prefix}.path must be a non-empty string")
        else:
            normalized_path = os.path.normcase(os.path.normpath(source.path))
            prior_id = seen_paths.get(normalized_path)
            if prior_id is not None and not manifest.allow_duplicate_paths:
                errors.append(
                    f"duplicate source path {source.path!r} is used by "
                    f"{prior_id!r} and {source.id!r}; set "
                    "allow_duplicate_paths=true only when intentional"
                )
            else:
                seen_paths[normalized_path] = source.id

        if source.tokenizer_path != manifest.tokenizer_path:
            errors.append(
                f"{prefix}.tokenizer_path {source.tokenizer_path!r} does not "
                f"exactly match manifest.tokenizer_path "
                f"{manifest.tokenizer_path!r}"
            )
        if isinstance(source.token_count, bool) or not isinstance(
            source.token_count, int
        ) or source.token_count < 0:
            errors.append(f"{prefix}.token_count must be a non-negative integer")
        if isinstance(source.weight, bool) or not isinstance(
            source.weight, (int, float)
        ) or not math.isfinite(source.weight) or source.weight <= 0:
            errors.append(f"{prefix}.weight must be a finite number greater than 0")
        else:
            total_weight += float(source.weight)
        if not _is_non_empty_string(source.license):
            errors.append(f"{prefix}.license must be provided")
        if not _is_valid_url(source.source_url):
            errors.append(
                f"{prefix}.source_url must be an absolute http(s) URL"
            )
        if not _is_non_empty_string(source.dataset_revision):
            errors.append(f"{prefix}.dataset_revision must be non-empty")
        if not isinstance(source.attribution_required, bool):
            errors.append(f"{prefix}.attribution_required must be boolean")
        if not isinstance(source.commercial_use_allowed, bool):
            errors.append(f"{prefix}.commercial_use_allowed must be boolean")
        if not _is_non_empty_string(source.split):
            errors.append(f"{prefix}.split must be a non-empty string")
        if source.content_hash is not None and (
            not isinstance(source.content_hash, str)
            or not SHA256_PATTERN.fullmatch(source.content_hash)
        ):
            errors.append(
                f"{prefix}.content_hash must be null or a 64-character SHA-256"
            )
        if not _is_non_empty_string(source.deduplication_status):
            errors.append(f"{prefix}.deduplication_status must be non-empty")
        if not isinstance(source.quality_filters, tuple) or any(
            not _is_non_empty_string(item) for item in source.quality_filters
        ):
            errors.append(
                f"{prefix}.quality_filters must contain only non-empty strings"
            )
        if not isinstance(source.notes, str):
            errors.append(f"{prefix}.notes must be a string")

    if manifest.sources and not math.isclose(
        total_weight, 1.0, rel_tol=0.0, abs_tol=weight_tolerance
    ):
        errors.append(
            f"source weights sum to {total_weight:.12g}; expected 1.0 "
            f"within absolute tolerance {weight_tolerance}"
        )

    if not errors:
        allocations = calculate_token_allocations(manifest)
        for source in manifest.sources:
            allocated = allocations[source.id]
            if allocated > source.token_count and not manifest.sampling_with_replacement:
                errors.append(
                    f"source {source.id!r} has token_count={source.token_count} "
                    f"but its weighted allocation requires {allocated}; add data, "
                    "lower its weight/target, or explicitly enable "
                    "sampling_with_replacement"
                )
            if allocated > 0 and source.token_count == 0:
                errors.append(
                    f"source {source.id!r} has a positive allocation but zero "
                    "available tokens, so replacement sampling is impossible"
                )

    if errors:
        raise MixtureValidationError(errors)
