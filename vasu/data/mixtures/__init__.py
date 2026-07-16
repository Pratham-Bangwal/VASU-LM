"""Public data-mixture planning and validation API."""

from .manifest import (
    load_manifest,
    manifest_from_dict,
    manifest_to_dict,
    resolve_source_path,
)
from .sampler import build_sampling_plan, iter_weighted_source_ids
from .schemas import (
    MixtureManifest,
    MixtureSource,
    SamplingPlan,
    SourceAllocation,
    SUPPORTED_DOMAINS,
    SUPPORTED_FORMATS,
)
from .validation import (
    MixtureValidationError,
    calculate_token_allocations,
    validate_manifest,
)

__all__ = [
    "MixtureManifest",
    "MixtureSource",
    "MixtureValidationError",
    "SamplingPlan",
    "SourceAllocation",
    "SUPPORTED_DOMAINS",
    "SUPPORTED_FORMATS",
    "build_sampling_plan",
    "calculate_token_allocations",
    "iter_weighted_source_ids",
    "load_manifest",
    "manifest_from_dict",
    "manifest_to_dict",
    "resolve_source_path",
    "validate_manifest",
]
