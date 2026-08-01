from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_base_v2_inventory import (
    validate_inventory_manifest_files,
)
from evaluation.framework.vasu_140m_base_v2_inventory_builder import (
    AUTHORING_SCHEMA_ID,
    SUPPORTED_DIMENSIONS,
    build_fixture_inventory,
    build_fixture_report,
    build_item_records,
)
from evaluation.framework.vasu_140m_base_v2_inventory_plan import plan_identity


COMMIT = "c" * 40
ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "configs/evaluation/vasu_140m_base_v2_inventory_construction_plan.json"
)


def construction_plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _provenance() -> dict[str, object]:
    return {
        "source_name": "VASU fixture source",
        "source_url": "https://example.com/source",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "fixture-revision-1",
        "citation": "Project-authored fixture for schema qualification.",
        "retrieved_at": "2026-08-01T12:00:00+05:30",
        "human_authored": True,
    }


def _item(dimension: str, ordinal: int = 1) -> dict[str, object]:
    item_id = f"{dimension}-fixture-{ordinal:03d}"
    family = f"{dimension}-family-{ordinal:03d}"
    common: dict[str, object] = {
        "schema_id": AUTHORING_SCHEMA_ID,
        "item_id": item_id,
        "dimension": dimension,
        "semantic_family_id": family,
        "parent_document_id": f"{dimension}-parent-{ordinal:03d}",
        "provenance": _provenance(),
        "human_approved": True,
    }
    if dimension == "factuality":
        common.update(
            strata={"semantic_family": family, "task_family": "science"},
            content={
                "prompt": "Which planet is known as the Red Planet?",
                "choices": [
                    {"choice_id": "a", "text": "Mars"},
                    {"choice_id": "b", "text": "Venus"},
                ],
            },
            scoring={"correct_choice_id": "a"},
        )
    elif dimension == "arithmetic":
        common.update(
            strata={
                "semantic_family": family,
                "operation": "addition",
                "difficulty": "fixture",
                "template_family": "fixture-direct-v1",
            },
            content={"prompt": "Calculate exactly: 1203 + 998 = ?"},
            scoring={"answer_type": "integer", "expected_answer": "2201"},
        )
    elif dimension == "repetition":
        common.update(
            strata={"semantic_family": family, "prompt_family": "narrative-prefix"},
            content={"prompt": "At the edge of the quiet forest, the old path continued"},
            scoring={"loop_ngram_size": 3},
        )
    elif dimension == "robustness":
        common.update(
            strata={
                "semantic_family": family,
                "variant_kind": "whitespace_and_case",
            },
            content={
                "baseline_prompt": "The capital of France is",
                "variant_prompt": "the   CAPITAL of France is",
            },
            scoring={"accepted_answers": ["Paris"]},
        )
    elif dimension == "manual_review":
        common.update(
            strata={"semantic_family": family, "category": "coherence"},
            content={"prompt": "A careful explanation of rainfall begins with"},
            scoring={
                "rubric_dimensions": [
                    "coherence",
                    "factual_support",
                    "degeneration",
                ]
            },
        )
    else:
        raise AssertionError(dimension)
    return common


def _repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    scorer_path = tmp_path / "evaluation/framework/scorer.py"
    scorer_path.parent.mkdir(parents=True)
    scorer_path.write_text("# fixture scorer\n", encoding="utf-8")
    return tmp_path, {
        "path": "evaluation/framework/scorer.py",
        "sha256": hashlib.sha256(scorer_path.read_bytes()).hexdigest(),
    }


def _build(tmp_path: Path, dimension: str, items: list[dict[str, object]] | None = None):
    root, scorer = _repo(tmp_path)
    output_parent = root / "evaluation/fixtures/builder"
    output_parent.mkdir(parents=True)
    return root, build_fixture_inventory(
        repository_root=root,
        output_directory=f"evaluation/fixtures/builder/{dimension}",
        inventory_id=f"fixture-{dimension}-v1",
        suite_id=str(construction_plan()["suite_id"]),
        repository_commit=str(construction_plan()["repository_commit"]),
        dimension=dimension,
        scorer=scorer,
        authoring_items=items or [_item(dimension)],
    )


@pytest.mark.parametrize("dimension", sorted(SUPPORTED_DIMENSIONS))
def test_builds_strict_development_bundle(tmp_path: Path, dimension: str) -> None:
    root, manifest = _build(tmp_path, dimension)
    validate_inventory_manifest_files(manifest, root)
    assert manifest["fixture_only"] is True
    assert manifest["production_suite_frozen"] is False
    assert manifest["evaluation_run_authorized"] is False
    assert manifest["training_authorized"] is False
    assert json.loads(
        (root / f"evaluation/fixtures/builder/{dimension}/manifest.json").read_text(
            encoding="utf-8"
        )
    ) == manifest


def test_report_requires_all_five_dimensions(tmp_path: Path) -> None:
    manifests = []
    for dimension in sorted(SUPPORTED_DIMENSIONS):
        dimension_root = tmp_path / dimension
        dimension_root.mkdir()
        _, manifest = _build(dimension_root, dimension)
        manifests.append(manifest)
    report = build_fixture_report(manifests, construction_plan=construction_plan())
    assert report["dimensions"] == sorted(SUPPORTED_DIMENSIONS)
    assert report["record_counts"] == {
        dimension: 1 for dimension in SUPPORTED_DIMENSIONS
    }
    assert report["production_inventory_created"] is False
    assert report["held_out_content_created"] is False
    assert report["encryption_key_created"] is False
    assert report["training_authorized"] is False
    assert report["construction_plan_sha256"] == construction_plan()["plan_sha256"]


def test_report_rejects_plan_not_bound_to_manifests(tmp_path: Path) -> None:
    manifests = []
    for dimension in sorted(SUPPORTED_DIMENSIONS):
        dimension_root = tmp_path / dimension
        dimension_root.mkdir()
        _, manifest = _build(dimension_root, dimension)
        manifests.append(manifest)
    value = construction_plan()
    value["suite_id"] = "substituted-suite"
    value["plan_sha256"] = plan_identity(value)
    with pytest.raises(ValueError, match="suite does not match plan"):
        build_fixture_report(manifests, construction_plan=value)


def test_rejects_unapproved_authoring() -> None:
    item = _item("arithmetic")
    item["human_approved"] = False
    with pytest.raises(ValueError, match="human approval"):
        build_item_records(item)


def test_rejects_semantic_family_mismatch() -> None:
    item = _item("repetition")
    item["strata"]["semantic_family"] = "substituted-family"
    with pytest.raises(ValueError, match="exact semantic family"):
        build_item_records(item)


def test_rejects_duplicate_prompt_identity(tmp_path: Path) -> None:
    first = _item("arithmetic", 1)
    second = _item("arithmetic", 2)
    second["content"] = copy.deepcopy(first["content"])
    with pytest.raises(ValueError, match="reuses a prompt"):
        _build(tmp_path, "arithmetic", [first, second])


def test_rejects_duplicate_parent_or_family(tmp_path: Path) -> None:
    first = _item("manual_review", 1)
    second = _item("manual_review", 2)
    second["content"] = {
        "prompt": "A concise account of evaporation should begin with"
    }
    second["parent_document_id"] = first["parent_document_id"]
    with pytest.raises(ValueError, match="parent documents"):
        _build(tmp_path, "manual_review", [first, second])


def test_refuses_existing_or_stale_output(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path, "factuality")
    scorer = manifest["scorer"]
    with pytest.raises(FileExistsError, match="already exists"):
        build_fixture_inventory(
            repository_root=root,
            output_directory="evaluation/fixtures/builder/factuality",
            inventory_id="fixture-factuality-v1",
            suite_id="suite",
            repository_commit=COMMIT,
            dimension="factuality",
            scorer=scorer,
            authoring_items=[_item("factuality")],
        )


def test_rejects_scorer_identity_mismatch(tmp_path: Path) -> None:
    root, scorer = _repo(tmp_path)
    (root / "evaluation/fixtures").mkdir()
    scorer["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scorer byte identity"):
        build_fixture_inventory(
            repository_root=root,
            output_directory="evaluation/fixtures/output",
            inventory_id="fixture-arithmetic-v1",
            suite_id="suite",
            repository_commit=COMMIT,
            dimension="arithmetic",
            scorer=scorer,
            authoring_items=[_item("arithmetic")],
        )


def test_tampered_payload_fails_bound_file_validation(tmp_path: Path) -> None:
    root, manifest = _build(tmp_path, "robustness")
    payload_path = root / str(manifest["payload"]["path"])
    with payload_path.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError, match="file identity"):
        validate_inventory_manifest_files(manifest, root)


def test_rejects_linked_output_parent_when_supported(tmp_path: Path) -> None:
    root, scorer = _repo(tmp_path)
    real_parent = root / "real-fixtures"
    real_parent.mkdir()
    linked_parent = root / "linked-fixtures"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(ValueError, match="link or junction"):
        build_fixture_inventory(
            repository_root=root,
            output_directory="linked-fixtures/output",
            inventory_id="fixture-arithmetic-v1",
            suite_id="suite",
            repository_commit=COMMIT,
            dimension="arithmetic",
            scorer=scorer,
            authoring_items=[_item("arithmetic")],
        )


@pytest.mark.parametrize("unsafe", ["../outside", "C:/absolute/output"])
def test_rejects_unsafe_output_path(tmp_path: Path, unsafe: str) -> None:
    root, scorer = _repo(tmp_path)
    with pytest.raises(ValueError, match="safe repository-relative"):
        build_fixture_inventory(
            repository_root=root,
            output_directory=unsafe,
            inventory_id="fixture-arithmetic-v1",
            suite_id="suite",
            repository_commit=COMMIT,
            dimension="arithmetic",
            scorer=scorer,
            authoring_items=[_item("arithmetic")],
        )
