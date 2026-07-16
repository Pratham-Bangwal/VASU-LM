from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from vasu.data.mixtures import (
    MixtureManifest,
    MixtureSource,
    MixtureValidationError,
    build_sampling_plan,
    iter_weighted_source_ids,
    load_manifest,
    manifest_from_dict,
    manifest_to_dict,
    resolve_source_path,
    validate_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def make_source(source_id: str = "source_a", **changes: object) -> MixtureSource:
    values: dict[str, object] = {
        "id": source_id,
        "domain": "general",
        "path": f"planned/{source_id}.jsonl",
        "format": "jsonl",
        "tokenizer_path": "assets/tokenizer.json",
        "token_count": 1_000,
        "weight": 1.0,
        "license": "MIT",
        "source_url": "https://example.com/datasets/source",
        "dataset_revision": "revision-1",
        "attribution_required": True,
        "commercial_use_allowed": True,
        "split": "train",
        "content_hash": "a" * 64,
        "deduplication_status": "exact duplicates removed",
        "quality_filters": ("non-empty text",),
        "notes": "Synthetic test source.",
    }
    values.update(changes)
    return MixtureSource(**values)  # type: ignore[arg-type]


def make_manifest(
    sources: tuple[MixtureSource, ...] | None = None,
    **changes: object,
) -> MixtureManifest:
    values: dict[str, object] = {
        "manifest_version": "1.0",
        "experiment_id": "test_mixture",
        "description": "Synthetic mixture test.",
        "tokenizer_path": "assets/tokenizer.json",
        "target_tokens": 100,
        "random_seed": 42,
        "sources": sources if sources is not None else (make_source(),),
        "created_at": "2026-07-16T22:00:00+05:30",
        "notes": "Test manifest.",
        "sampling_with_replacement": False,
        "allow_duplicate_paths": False,
    }
    values.update(changes)
    return MixtureManifest(**values)  # type: ignore[arg-type]


def assert_invalid(manifest: MixtureManifest, message: str) -> None:
    with pytest.raises(MixtureValidationError, match=message):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    "config_name",
    [
        "vasu_60m_control_pilot.json",
        "vasu_60m_factual_pilot.json",
        "vasu_60m_capability_pilot.json",
    ],
)
def test_pilot_configs_are_valid_and_allocate_exact_targets(
    config_name: str,
) -> None:
    manifest = load_manifest(REPOSITORY_ROOT / "configs" / "data" / config_name)
    validate_manifest(manifest)
    plan = build_sampling_plan(manifest)
    assert plan.allocated_tokens == manifest.target_tokens
    assert not plan.sampling_with_replacement


def test_manifest_loader_rejects_missing_invalid_and_unknown_fields(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_manifest(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{bad", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_manifest(invalid)

    payload = manifest_to_dict(make_manifest())
    del payload["experiment_id"]
    with pytest.raises(ValueError, match="missing required fields"):
        manifest_from_dict(payload)

    payload = manifest_to_dict(make_manifest())
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        manifest_from_dict(payload)


def test_source_loader_rejects_missing_fields_and_non_list_filters() -> None:
    payload = manifest_to_dict(make_manifest())
    del payload["sources"][0]["license"]
    with pytest.raises(ValueError, match="missing required fields"):
        manifest_from_dict(payload)

    payload = manifest_to_dict(make_manifest())
    payload["sources"][0]["quality_filters"] = "not-a-list"
    with pytest.raises(ValueError, match="must be a JSON list"):
        manifest_from_dict(payload)


def test_duplicate_ids_and_empty_sources_are_rejected() -> None:
    duplicate = make_manifest(
        (make_source("same", weight=0.5), make_source("same", weight=0.5))
    )
    assert_invalid(duplicate, "duplicate source id")
    assert_invalid(make_manifest(()), "at least one source")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("token_count", -1, "non-negative integer"),
        ("weight", 0.0, "greater than 0"),
        ("license", "", "license must be provided"),
        ("dataset_revision", "", "dataset_revision must be non-empty"),
        ("commercial_use_allowed", "yes", "must be boolean"),
        ("attribution_required", 1, "must be boolean"),
        ("source_url", "not-a-url", "absolute http"),
        ("content_hash", "1234", "64-character SHA-256"),
        ("content_hash", 1234, "64-character SHA-256"),
        ("domain", "medical", "unsupported"),
        ("format", "csv", "unsupported"),
    ],
)
def test_invalid_source_values_are_actionable(
    field: str,
    value: object,
    message: str,
) -> None:
    assert_invalid(make_manifest((make_source(**{field: value}),)), message)


def test_manifest_level_values_are_strict() -> None:
    assert_invalid(make_manifest(target_tokens=0), "at least 1")
    assert_invalid(make_manifest(random_seed=-1), r"0 to 2\^63-1")
    assert_invalid(make_manifest(sampling_with_replacement="true"), "must be boolean")
    assert_invalid(make_manifest(created_at="2026-07-16"), "timezone offset")
    assert_invalid(make_manifest(manifest_version="2.0"), "unsupported")


def test_weights_must_sum_to_one_without_silent_normalization() -> None:
    sources = (
        make_source("a", weight=0.4),
        make_source("b", weight=0.5),
    )
    assert_invalid(make_manifest(sources), "weights sum to 0.9")


def test_source_tokenizer_must_match_exactly() -> None:
    manifest = make_manifest(
        (make_source(tokenizer_path="./assets/tokenizer.json"),)
    )
    assert_invalid(manifest, "does not exactly match")


def test_duplicate_paths_require_explicit_permission() -> None:
    sources = (
        make_source("a", path="planned/data.jsonl", weight=0.5),
        make_source("b", path="planned/./data.jsonl", weight=0.5),
    )
    assert_invalid(make_manifest(sources), "duplicate source path")
    validate_manifest(make_manifest(sources, allow_duplicate_paths=True))


def test_allocation_capacity_requires_explicit_replacement() -> None:
    source = make_source(token_count=10)
    assert_invalid(make_manifest((source,), target_tokens=11), "requires 11")

    replacement = make_manifest(
        (source,),
        target_tokens=11,
        sampling_with_replacement=True,
    )
    plan = build_sampling_plan(replacement)
    assert plan.allocation_for("source_a").replacement_required

    zero = make_manifest(
        (make_source(token_count=0),),
        sampling_with_replacement=True,
    )
    assert_invalid(zero, "replacement sampling is impossible")


def test_largest_remainder_allocation_is_exact_and_deterministic() -> None:
    sources = (
        make_source("a", weight=1 / 3),
        make_source("b", weight=1 / 3),
        make_source("c", weight=1 / 3),
    )
    plan = build_sampling_plan(make_manifest(sources, target_tokens=10))
    allocations = {
        item.source_id: item.allocated_tokens for item in plan.allocations
    }
    assert allocations == {"a": 4, "b": 3, "c": 3}
    assert plan.allocated_tokens == 10


def test_sampling_plan_and_weighted_draws_are_reproducible() -> None:
    sources = (
        make_source("a", weight=0.25),
        make_source("b", weight=0.75),
    )
    manifest = make_manifest(sources, random_seed=123)
    assert build_sampling_plan(manifest) == build_sampling_plan(manifest)
    first = list(iter_weighted_source_ids(manifest, 100))
    second = list(iter_weighted_source_ids(manifest, 100))
    changed = list(iter_weighted_source_ids(replace(manifest, random_seed=124), 100))
    assert first == second
    assert first != changed
    assert set(first) <= {"a", "b"}


def test_draw_count_is_validated() -> None:
    manifest = make_manifest()
    with pytest.raises(TypeError, match="integer"):
        list(iter_weighted_source_ids(manifest, True))
    with pytest.raises(ValueError, match="non-negative"):
        list(iter_weighted_source_ids(manifest, -1))


def test_path_resolution_requires_an_explicit_base_and_not_file_existence(
    tmp_path: Path,
) -> None:
    source = make_source(path="future/source.jsonl", content_hash=None)
    resolved = resolve_source_path(source, tmp_path)
    assert resolved == tmp_path / "future" / "source.jsonl"
    assert not resolved.exists()
    validate_manifest(make_manifest((source,)))


def test_json_round_trip_preserves_manifest() -> None:
    manifest = make_manifest()
    payload = manifest_to_dict(manifest)
    assert isinstance(payload["sources"][0]["quality_filters"], list)
    assert manifest_from_dict(payload) == manifest
    assert manifest_from_dict(json.loads(json.dumps(payload))) == manifest
