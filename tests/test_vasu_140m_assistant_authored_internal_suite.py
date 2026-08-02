from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_assistant_authored_internal_suite import (
    COUNTS,
    SCORER_PATH,
    SUITE_ID,
    build_suite,
    generated_authoring_items,
)
from evaluation.framework.vasu_140m_base_v2_inventory import (
    validate_inventory_manifest_files,
)


COMMIT = "a" * 40


def _root(tmp_path: Path) -> Path:
    scorer = tmp_path / SCORER_PATH
    scorer.parent.mkdir(parents=True)
    scorer.write_text("# internal suite test scorer\n", encoding="utf-8")
    (tmp_path / "evaluation/fixtures").mkdir(parents=True)
    return tmp_path


def test_generated_items_have_expected_counts_and_nonhuman_provenance() -> None:
    records = generated_authoring_items()
    assert {name: len(values) for name, values in records.items()} == COUNTS
    for values in records.values():
        assert all(item["provenance"]["human_authored"] is False for item in values)


def test_builds_complete_non_authorizing_internal_fixture(tmp_path: Path) -> None:
    root = _root(tmp_path)
    report = build_suite(
        repository_root=root,
        output_directory="evaluation/fixtures/internal",
        repository_commit=COMMIT,
    )
    assert report["suite_id"] == SUITE_ID
    assert report["record_counts"] == COUNTS
    assert report["authoring"]["independently_curated"] is False
    assert report["authoring"]["held_out_content_present"] is False
    assert report["authoring"]["training_data_eligible"] is False
    for dimension in COUNTS:
        manifest = json.loads(
            (root / f"evaluation/fixtures/internal/{dimension}/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        validate_inventory_manifest_files(manifest, root)
        assert manifest["fixture_only"] is True
        assert manifest["training_authorized"] is False


def test_refuses_to_overwrite_internal_fixture(tmp_path: Path) -> None:
    root = _root(tmp_path)
    kwargs = {
        "repository_root": root,
        "output_directory": "evaluation/fixtures/internal",
        "repository_commit": COMMIT,
    }
    build_suite(**kwargs)
    with pytest.raises(FileExistsError, match="already exists"):
        build_suite(**kwargs)


def test_rejects_unsafe_output_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        build_suite(
            repository_root=_root(tmp_path),
            output_directory="../outside",
            repository_commit=COMMIT,
        )
