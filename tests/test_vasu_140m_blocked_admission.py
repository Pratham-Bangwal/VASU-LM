from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vasu.data.vasu_140m_blocked_admission import (
    evidence_identity, validate_blocked_admission_files,
)


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "configs/data/admissions/blocked_evidence"


def test_blocked_admission_evidence_is_bound_and_non_authorizing() -> None:
    paths = sorted(DIRECTORY.glob("*.blocked.json"))
    assert len(paths) == 2
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_blocked_admission_files(value, ROOT, require_external_artifacts=False)
        assert value["decision"] == "blocked"
        assert value["training_authorized"] is False


def test_blocked_admission_cannot_drop_independent_gate() -> None:
    path = next(DIRECTORY.glob("*.blocked.json"))
    value = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(value)
    changed["remaining_blockers"].pop()
    changed["evidence_sha256"] = evidence_identity(changed)
    with pytest.raises(ValueError, match="remaining blockers"):
        validate_blocked_admission_files(
            changed, ROOT, require_external_artifacts=False
        )
