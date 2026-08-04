from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_development_promotion import (
    DEPENDENCIES,
    DIMENSIONS,
    OUTPUT_ROOT,
    SOURCE_ROOT,
    promote_development_inventories,
)


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for path in (
        SOURCE_ROOT,
        Path("evaluation/framework"),
        Path("evaluation/candidates"),
    ):
        (root / path).mkdir(parents=True, exist_ok=True)
    real = Path.cwd()
    for dimension in DIMENSIONS:
        shutil.copytree(real / SOURCE_ROOT / dimension, root / SOURCE_ROOT / dimension)
    scorer = real / "evaluation/framework/vasu_140m_base_v2_tasks.py"
    shutil.copyfile(scorer, root / "evaluation/framework/vasu_140m_base_v2_tasks.py")
    for path in DEPENDENCIES:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(real / path, root / path)
    return root


def test_promotes_all_five_without_changing_record_bytes(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    receipt = promote_development_inventories(root, "a" * 40)
    assert set(receipt["promoted_inventory_sha256s"]) == set(DIMENSIONS)
    for dimension in DIMENSIONS:
        for filename in ("payload.jsonl", "provenance.jsonl", "contamination.jsonl"):
            assert (root / SOURCE_ROOT / dimension / filename).read_bytes() == (
                root / OUTPUT_ROOT / dimension / filename
            ).read_bytes()
    assert receipt["production_suite_frozen"] is False
    assert receipt["training_authorized"] is False


def test_refuses_overwrite(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    (root / OUTPUT_ROOT).mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        promote_development_inventories(root, "a" * 40)


def test_rejects_invalid_commit_before_writes(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    with pytest.raises(ValueError, match="Git commit"):
        promote_development_inventories(root, "invalid")
    assert not (root / OUTPUT_ROOT).exists()


def test_rejects_dependency_mutation(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    dependency = root / next(iter(DEPENDENCIES))
    dependency.write_text("mutation", encoding="utf-8")
    with pytest.raises(ValueError, match="dependency identity mismatch"):
        promote_development_inventories(root, "a" * 40)
