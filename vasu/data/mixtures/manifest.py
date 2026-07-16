"""Explicit JSON serialization for data-mixture manifests."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import MixtureManifest, MixtureSource


SOURCE_FIELDS = frozenset(MixtureSource.__dataclass_fields__)
MANIFEST_REQUIRED_FIELDS = frozenset(
    {
        "manifest_version",
        "experiment_id",
        "description",
        "tokenizer_path",
        "target_tokens",
        "random_seed",
        "sources",
        "created_at",
        "notes",
    }
)
MANIFEST_OPTIONAL_FIELDS = frozenset(
    {"sampling_with_replacement", "allow_duplicate_paths"}
)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _check_fields(
    value: Mapping[str, Any],
    required: frozenset[str],
    optional: frozenset[str],
    location: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise ValueError(f"{location} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{location} has unknown fields: {unknown}")


def source_from_dict(payload: Mapping[str, Any], index: int) -> MixtureSource:
    location = f"sources[{index}]"
    value = _require_mapping(payload, location)
    _check_fields(value, SOURCE_FIELDS, frozenset(), location)

    quality_filters = value["quality_filters"]
    if not isinstance(quality_filters, list):
        raise ValueError(f"{location}.quality_filters must be a JSON list")

    try:
        return MixtureSource(
            id=value["id"],
            domain=value["domain"],
            path=value["path"],
            format=value["format"],
            tokenizer_path=value["tokenizer_path"],
            token_count=value["token_count"],
            weight=value["weight"],
            license=value["license"],
            source_url=value["source_url"],
            dataset_revision=value["dataset_revision"],
            attribution_required=value["attribution_required"],
            commercial_use_allowed=value["commercial_use_allowed"],
            split=value["split"],
            content_hash=value["content_hash"],
            deduplication_status=value["deduplication_status"],
            quality_filters=tuple(quality_filters),
            notes=value["notes"],
        )
    except TypeError as error:
        raise ValueError(f"Invalid {location}: {error}") from error


def manifest_from_dict(payload: Mapping[str, Any]) -> MixtureManifest:
    value = _require_mapping(payload, "manifest")
    _check_fields(
        value,
        MANIFEST_REQUIRED_FIELDS,
        MANIFEST_OPTIONAL_FIELDS,
        "manifest",
    )
    raw_sources = value["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError("manifest.sources must be a JSON list")
    sources = tuple(
        source_from_dict(source, index)
        for index, source in enumerate(raw_sources)
    )
    return MixtureManifest(
        manifest_version=value["manifest_version"],
        experiment_id=value["experiment_id"],
        description=value["description"],
        tokenizer_path=value["tokenizer_path"],
        target_tokens=value["target_tokens"],
        random_seed=value["random_seed"],
        sources=sources,
        created_at=value["created_at"],
        notes=value["notes"],
        sampling_with_replacement=value.get(
            "sampling_with_replacement", False
        ),
        allow_duplicate_paths=value.get("allow_duplicate_paths", False),
    )


def load_manifest(path: str | Path) -> MixtureManifest:
    """Load and validate a JSON mixture manifest."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Mixture manifest does not exist: {manifest_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Mixture manifest is not valid JSON at line {error.lineno}, "
            f"column {error.colno}: {manifest_path}"
        ) from error
    manifest = manifest_from_dict(payload)
    # Local import avoids a module cycle while making the public file-loading
    # boundary strict: an invalid manifest is never returned to its caller.
    from .validation import validate_manifest

    validate_manifest(manifest)
    return manifest


def manifest_to_dict(manifest: MixtureManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    sources: list[dict[str, Any]] = []
    for source in manifest.sources:
        source_payload = asdict(source)
        source_payload["quality_filters"] = list(source.quality_filters)
        sources.append(source_payload)
    payload["sources"] = sources
    return payload


def resolve_source_path(source: MixtureSource, base_directory: str | Path) -> Path:
    """Resolve a source only against a caller-provided directory."""
    path = Path(source.path)
    return path if path.is_absolute() else Path(base_directory) / path
