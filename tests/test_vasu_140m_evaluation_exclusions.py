from __future__ import annotations

import json
from pathlib import Path

import pytest

from vasu.data.vasu_140m_evaluation_exclusions import (
    registry_identity,
    validate_exclusion_registry_files,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/data/exclusions/vasu_140m_likelihood_self_curated_v1.json"


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_exclusion_registry_binds_exact_likelihood_parents() -> None:
    registry = _registry()
    validate_exclusion_registry_files(registry, ROOT)
    assert registry["fixture_only"] is True
    assert registry["independently_curated"] is False
    assert registry["training_authorized"] is False
    assert [source["excluded_parent_count"] for source in registry["sources"]] == [1024, 1024]


def test_exclusion_registry_rejects_removed_parent() -> None:
    registry = _registry()
    registry["sources"][0]["excluded_parent_document_ids"].pop()
    registry["registry_sha256"] = registry_identity(registry)
    with pytest.raises(ValueError, match="do not match"):
        validate_exclusion_registry_files(registry, ROOT)


def test_exclusion_registry_rejects_authorization_drift() -> None:
    registry = _registry()
    registry["training_authorized"] = True
    registry["registry_sha256"] = registry_identity(registry)
    with pytest.raises(ValueError, match="must remain false"):
        validate_exclusion_registry_files(registry, ROOT)
