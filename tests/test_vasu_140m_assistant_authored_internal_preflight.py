from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_assistant_authored_internal_preflight import (
    qualification_identity,
    qualify_internal_suite,
)
from evaluation.framework.vasu_140m_assistant_authored_internal_suite import COUNTS


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_validates_frozen_internal_suite_without_execution() -> None:
    report = qualify_internal_suite(ROOT)
    assert report["record_counts"] == COUNTS
    assert report["held_out_content_present"] is False
    assert report["checkpoint_opened"] is False
    assert report["model_invoked"] is False
    assert report["training_authorized"] is False
    assert report["qualification_sha256"] == qualification_identity(report)


def test_preflight_identity_changes_when_execution_boundary_changes() -> None:
    report = qualify_internal_suite(ROOT)
    changed = copy.deepcopy(report)
    changed["model_invoked"] = True
    assert qualification_identity(changed) != report["qualification_sha256"]
    with pytest.raises(ValueError):
        qualify_internal_suite(ROOT / "missing")
