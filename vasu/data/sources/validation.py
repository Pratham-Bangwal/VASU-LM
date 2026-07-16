"""Validation for source records and mixture-to-registry references."""

from __future__ import annotations

from datetime import datetime
import re
from urllib.parse import urlparse

from vasu.data.mixtures.schemas import MixtureManifest, SUPPORTED_DOMAINS

from .schemas import APPROVAL_STATUSES, DataSourceRecord, SourceRegistry


SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
UNRESOLVED_MARKERS = ("todo", "tbd", "unresolved", "unknown", "pending review")


class SourceRegistryValidationError(ValueError):
    """One or more actionable source-registry validation failures."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("Source registry validation failed:\n- " + "\n- ".join(errors))


def _non_empty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_url(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _valid_timestamp(value: object) -> bool:
    if not _non_empty(value):
        return False
    try:
        parsed = datetime.fromisoformat(value)  # type: ignore[arg-type]
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_string_list(
    value: object,
    field: str,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(value, tuple) or not value or any(
        not _non_empty(item) for item in value
    ):
        errors.append(f"{prefix}.{field} must contain at least one non-empty string")


def validate_source_record(record: DataSourceRecord) -> None:
    errors: list[str] = []
    prefix = f"source {record.source_id!r}"

    for field in (
        "source_id",
        "display_name",
        "provider",
        "dataset_name",
        "access_method",
        "subset",
        "split",
        "pinned_revision",
        "license_name",
        "intended_purpose",
    ):
        if not _non_empty(getattr(record, field)):
            errors.append(f"{prefix}.{field} must be a non-empty string")

    for field in ("dataset_card_url", "homepage_url", "license_url"):
        if not _valid_url(getattr(record, field)):
            errors.append(f"{prefix}.{field} must be an absolute http(s) URL")

    if not isinstance(record.approval_status, str) or (
        record.approval_status not in APPROVAL_STATUSES
    ):
        errors.append(
            f"{prefix}.approval_status {record.approval_status!r} is unsupported; "
            f"choose one of {sorted(APPROVAL_STATUSES)}"
        )

    policy_fields = (
        "commercial_use_allowed",
        "attribution_required",
        "redistribution_allowed",
        "gated_access",
        "requires_authentication",
    )
    for field in policy_fields:
        value = getattr(record, field)
        if value is not None and not isinstance(value, bool):
            errors.append(f"{prefix}.{field} must be boolean or null")
        elif value is None and record.approval_status not in {"pending", "blocked"}:
            errors.append(
                f"{prefix}.{field} may be null only for pending or blocked sources"
            )

    for field in (
        "expected_download_size_bytes",
        "expected_raw_examples",
        "expected_raw_tokens",
    ):
        value = getattr(record, field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            errors.append(f"{prefix}.{field} must be null or a non-negative integer")

    _validate_string_list(record.supported_domains, "supported_domains", prefix, errors)
    if isinstance(record.supported_domains, tuple):
        unsupported = sorted(
            domain
            for domain in record.supported_domains
            if isinstance(domain, str) and domain not in SUPPORTED_DOMAINS
        )
        if unsupported:
            errors.append(
                f"{prefix}.supported_domains contains unsupported domains: {unsupported}"
            )
    for field in (
        "quality_risks",
        "required_quality_filters",
        "deduplication_requirements",
        "contamination_risks",
    ):
        _validate_string_list(getattr(record, field), field, prefix, errors)

    for field in (
        "local_raw_path",
        "local_processed_path",
        "approval_notes",
        "reviewed_by",
        "reviewed_at",
        "notes",
    ):
        if not isinstance(getattr(record, field), str):
            errors.append(f"{prefix}.{field} must be a string")

    if record.content_hash is not None and (
        not isinstance(record.content_hash, str)
        or not SHA256_PATTERN.fullmatch(record.content_hash)
    ):
        errors.append(f"{prefix}.content_hash must be null or a 64-character SHA-256")

    if record.reviewed_at and not _valid_timestamp(record.reviewed_at):
        errors.append(f"{prefix}.reviewed_at must be an ISO-8601 timestamp with timezone")

    if record.approval_status == "approved":
        unresolved = record.approval_notes.lower() if isinstance(record.approval_notes, str) else ""
        if any(marker in unresolved for marker in UNRESOLVED_MARKERS):
            errors.append(f"{prefix}.approval_notes still contains unresolved review work")
        if not _non_empty(record.reviewed_by):
            errors.append(f"{prefix}.reviewed_by is required when approved")
        if not _valid_timestamp(record.reviewed_at):
            errors.append(
                f"{prefix}.reviewed_at is required with a timezone when approved"
            )
        if record.commercial_use_allowed is None:
            errors.append(f"{prefix}.commercial_use_allowed cannot be unknown when approved")
        if record.redistribution_allowed is None:
            errors.append(f"{prefix}.redistribution_allowed cannot be unknown when approved")
        if record.gated_access is True and not _non_empty(record.access_method):
            errors.append(f"{prefix}.access_method must document gated access")
        if not _non_empty(record.local_raw_path) or not _non_empty(
            record.local_processed_path
        ):
            errors.append(
                f"{prefix} must define local_raw_path and local_processed_path when approved"
            )

    if errors:
        raise SourceRegistryValidationError(errors)


def validate_registry(registry: SourceRegistry) -> None:
    errors: list[str] = []
    if not registry.records:
        errors.append("registry must contain at least one source record")
    seen: dict[str, int] = {}
    for index, record in enumerate(registry.records):
        try:
            validate_source_record(record)
        except SourceRegistryValidationError as error:
            errors.extend(error.errors)
        if _non_empty(record.source_id):
            if record.source_id in seen:
                errors.append(
                    f"duplicate source_id {record.source_id!r} at registry indexes "
                    f"{seen[record.source_id]} and {index}"
                )
            else:
                seen[record.source_id] = index
    if errors:
        raise SourceRegistryValidationError(errors)


def validate_manifest_sources(
    manifest: MixtureManifest,
    registry: SourceRegistry,
    require_approved: bool = False,
) -> None:
    """Validate source references and optionally require preparation approval."""
    if not isinstance(require_approved, bool):
        raise TypeError("require_approved must be boolean")
    validate_registry(registry)
    records = {record.source_id: record for record in registry.records}
    errors: list[str] = []
    for source in manifest.sources:
        record = records.get(source.id)
        if record is None:
            errors.append(
                f"manifest source {source.id!r} is absent from the source registry"
            )
            continue
        prefix = f"manifest source {source.id!r}"
        comparisons = (
            ("dataset_card_url", record.dataset_card_url, source.source_url),
            ("pinned_revision", record.pinned_revision, source.dataset_revision),
            ("license_name", record.license_name, source.license),
            ("split", record.split, source.split),
            ("local_processed_path", record.local_processed_path, source.path),
        )
        for field, registry_value, manifest_value in comparisons:
            if registry_value != manifest_value:
                errors.append(
                    f"{prefix} {field} mismatch: registry={registry_value!r}, "
                    f"manifest={manifest_value!r}"
                )
        if source.domain not in record.supported_domains:
            errors.append(
                f"{prefix} domain {source.domain!r} is not declared by the registry"
            )
        for field in ("commercial_use_allowed", "attribution_required"):
            policy = getattr(record, field)
            if policy is not None and policy != getattr(source, field):
                errors.append(f"{prefix} {field} conflicts with registry policy")
        if source.content_hash is not None and source.content_hash != record.content_hash:
            errors.append(f"{prefix} content_hash does not match the registry")
        if require_approved and record.approval_status != "approved":
            errors.append(
                f"{prefix} is not training-ready: approval_status="
                f"{record.approval_status!r}"
            )
    if errors:
        raise SourceRegistryValidationError(errors)

