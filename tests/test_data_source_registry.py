from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from vasu.data.mixtures import load_manifest
from vasu.data.sources import (
    DataSourceRecord,
    SourceRegistry,
    SourceRegistryValidationError,
    get_source,
    load_source_record,
    load_source_registry,
    source_record_to_dict,
    validate_manifest_sources,
    validate_registry,
    validate_source_record,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = ROOT / "configs" / "data" / "sources"


def make_record(source_id: str = "test_source", **changes: object) -> DataSourceRecord:
    values: dict[str, object] = {
        "source_id": source_id,
        "display_name": "Test Source",
        "provider": "Test Provider",
        "dataset_name": "provider/test",
        "dataset_card_url": "https://example.com/dataset-card",
        "homepage_url": "https://example.com/",
        "access_method": "Public HTTPS snapshot",
        "subset": "default",
        "split": "train",
        "pinned_revision": "abc123",
        "license_name": "MIT",
        "license_url": "https://opensource.org/license/mit",
        "commercial_use_allowed": True,
        "attribution_required": True,
        "redistribution_allowed": True,
        "gated_access": False,
        "requires_authentication": False,
        "expected_download_size_bytes": 100,
        "expected_raw_examples": 10,
        "expected_raw_tokens": 1_000,
        "supported_domains": ("general",),
        "intended_purpose": "Synthetic validation tests.",
        "quality_risks": ("synthetic risk",),
        "required_quality_filters": ("remove empty records",),
        "deduplication_requirements": ("exact deduplication",),
        "contamination_risks": ("possible benchmark overlap",),
        "local_raw_path": "data/raw/test",
        "local_processed_path": "data/processed/test.jsonl",
        "approval_status": "approved",
        "approval_notes": "Approved for synthetic testing.",
        "reviewed_by": "Test Reviewer",
        "reviewed_at": "2026-07-16T22:00:00+05:30",
        "content_hash": "a" * 64,
        "notes": "Synthetic record.",
    }
    values.update(changes)
    return DataSourceRecord(**values)  # type: ignore[arg-type]


def assert_invalid(record: DataSourceRecord, message: str) -> None:
    with pytest.raises(SourceRegistryValidationError, match=message):
        validate_source_record(record)


def test_repository_registry_loads_all_mixture_source_ids() -> None:
    registry = load_source_registry(SOURCE_DIRECTORY)
    assert set(registry.source_ids) == {
        "fineweb_edu_original_train",
        "fineweb_edu_extension_2025_26",
        "wikipedia_en_20231101_planned",
        "finemath_4plus_planned",
        "permissive_python_code_planned",
        "vasu_verified_reasoning_v1_planned",
    }


def test_repository_manifests_match_registry_metadata() -> None:
    registry = load_source_registry(SOURCE_DIRECTORY)
    for path in sorted((ROOT / "configs" / "data").glob("vasu_60m_*_pilot.json")):
        validate_manifest_sources(load_manifest(path), registry)


def test_control_and_factual_manifests_are_registry_ready() -> None:
    registry = load_source_registry(SOURCE_DIRECTORY)
    for name in (
        "vasu_60m_control_pilot.json",
        "vasu_60m_factual_pilot.json",
    ):
        manifest = load_manifest(ROOT / "configs" / "data" / name)
        validate_manifest_sources(manifest, registry, require_approved=True)

    capability = load_manifest(
        ROOT / "configs/data/vasu_60m_capability_pilot.json"
    )
    with pytest.raises(SourceRegistryValidationError, match="not training-ready"):
        validate_manifest_sources(capability, registry, require_approved=True)


def test_load_single_record_and_family_bundle() -> None:
    record = load_source_record(SOURCE_DIRECTORY / "wikimedia.json")
    assert record.source_id == "wikipedia_en_20231101_planned"
    assert record.approval_status == "approved"
    assert record.pinned_revision.startswith("e6057dc557255a03")
    assert record.expected_download_size_bytes == 11_630_929_031
    assert record.expected_raw_examples == 6_407_814
    with pytest.raises(ValueError, match="Expected one source-record"):
        load_source_record(SOURCE_DIRECTORY / "fineweb_edu.json")
    registry = load_source_registry(SOURCE_DIRECTORY)
    assert get_source(registry, "fineweb_edu_extension_2025_26").approval_status == "approved"


def test_loading_errors_are_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_source_record(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_source_record(invalid)
    with pytest.raises(FileNotFoundError, match="directory does not exist"):
        load_source_registry(tmp_path / "missing-directory")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no JSON records"):
        load_source_registry(empty)


def test_missing_and_unknown_record_fields_are_rejected(tmp_path: Path) -> None:
    payload = source_record_to_dict(make_record())
    del payload["provider"]
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_source_record(path)

    payload = source_record_to_dict(make_record())
    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_source_record(path)


def test_duplicate_source_ids_are_rejected() -> None:
    registry = SourceRegistry((make_record("same"), make_record("same")))
    with pytest.raises(SourceRegistryValidationError, match="duplicate source_id"):
        validate_registry(registry)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("approval_status", "ready", "unsupported"),
        ("provider", "", "provider must be a non-empty"),
        ("dataset_name", "", "dataset_name must be a non-empty"),
        ("pinned_revision", "", "pinned_revision must be a non-empty"),
        ("license_name", "", "license_name must be a non-empty"),
        ("dataset_card_url", "bad", "absolute http"),
        ("homepage_url", "ftp://example.com", "absolute http"),
        ("license_url", "", "absolute http"),
        ("commercial_use_allowed", "yes", "boolean or null"),
        ("attribution_required", 1, "boolean or null"),
        ("expected_download_size_bytes", -1, "non-negative integer"),
        ("expected_raw_examples", -1, "non-negative integer"),
        ("expected_raw_tokens", -1, "non-negative integer"),
        ("supported_domains", (), "at least one non-empty"),
        ("supported_domains", ("medical",), "unsupported domains"),
        ("intended_purpose", "", "intended_purpose must be a non-empty"),
        ("quality_risks", (), "at least one non-empty"),
        ("required_quality_filters", (), "at least one non-empty"),
        ("deduplication_requirements", (), "at least one non-empty"),
        ("content_hash", "bad", "64-character SHA-256"),
    ],
)
def test_invalid_record_values_are_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    assert_invalid(make_record(**{field: value}), message)


def test_null_policy_values_are_only_allowed_for_pending_or_blocked() -> None:
    pending = make_record(
        approval_status="pending",
        commercial_use_allowed=None,
        attribution_required=None,
        redistribution_allowed=None,
        gated_access=None,
        requires_authentication=None,
        reviewed_by="",
        reviewed_at="",
    )
    validate_source_record(pending)
    assert_invalid(
        replace(pending, approval_status="rejected"),
        "may be null only for pending or blocked",
    )


def test_approved_record_requires_resolved_review_and_local_path_plan() -> None:
    assert_invalid(make_record(approval_notes="TODO: review license"), "unresolved")
    assert_invalid(make_record(reviewed_by=""), "reviewed_by is required")
    assert_invalid(make_record(reviewed_at=""), "reviewed_at is required")
    assert_invalid(make_record(redistribution_allowed=None), "cannot be unknown")
    assert_invalid(make_record(commercial_use_allowed=None), "cannot be unknown")
    assert_invalid(make_record(local_processed_path=""), "must define local_raw_path")


def test_review_timestamp_requires_timezone() -> None:
    assert_invalid(make_record(reviewed_at="2026-07-16"), "with timezone")


def test_get_source_unknown_id_lists_available_ids() -> None:
    registry = SourceRegistry((make_record("known"),))
    with pytest.raises(KeyError, match="available source IDs: known"):
        get_source(registry, "missing")


def test_manifest_source_must_exist_and_metadata_must_match() -> None:
    manifest = load_manifest(ROOT / "configs/data/vasu_60m_control_pilot.json")
    registry = load_source_registry(SOURCE_DIRECTORY)
    without_original = SourceRegistry(
        tuple(
            record
            for record in registry.records
            if record.source_id != "fineweb_edu_original_train"
        )
    )
    with pytest.raises(SourceRegistryValidationError, match="absent from"):
        validate_manifest_sources(manifest, without_original)

    original = get_source(registry, "fineweb_edu_original_train")
    changed = replace(original, pinned_revision="wrong-revision")
    mismatched = SourceRegistry(
        tuple(changed if record.source_id == changed.source_id else record for record in registry.records)
    )
    with pytest.raises(SourceRegistryValidationError, match="pinned_revision mismatch"):
        validate_manifest_sources(manifest, mismatched)


def test_source_record_round_trip_is_json_compatible(tmp_path: Path) -> None:
    record = make_record()
    payload = source_record_to_dict(record)
    assert isinstance(payload["supported_domains"], list)
    path = tmp_path / "record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_source_record(path) == record


def test_registry_cli_validates_factual_readiness(capsys: pytest.CaptureFixture[str]) -> None:
    from vasu.data.sources.registry import main

    result = main(
        [
            "--registry-dir",
            str(SOURCE_DIRECTORY),
            "--manifest",
            str(ROOT / "configs/data/vasu_60m_factual_pilot.json"),
            "--require-approved",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert "Validated source registry: 6 records" in output
    assert "vasu_60m_factual_pilot (approved, 3 sources)" in output
