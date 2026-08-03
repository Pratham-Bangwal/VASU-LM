from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_base_v2 import canonical_json
from vasu.data.vasu_140m_semantic_quarantine import build_quarantine, write_quarantine


def _review(path: Path) -> Path:
    report = {
        "exact_candidates": [
            {
                "candidate_id": "exact-1",
                "decision": "quarantine_exact_prompt_or_answer_overlap",
                "source": "fineweb",
                "source_record_ordinal_sha256": hashlib.sha256(b"2").hexdigest(),
            },
            {
                "candidate_id": "exact-2",
                "decision": "quarantine_exact_prompt_or_answer_overlap",
                "source": "fineweb",
                "source_record_ordinal_sha256": hashlib.sha256(b"2").hexdigest(),
            },
        ],
        "held_out_content_exposed": False,
        "private_key_opened": False,
        "result_sha256": "",
        "schema_id": "vasu_140m_base_v2_semantic_contamination_independent_review_v1",
        "semantic_candidates": [
            {
                "candidate_id": "accepted",
                "decision": "accept_low_risk_generic_or_partial_overlap",
                "source": "fineweb",
                "source_record_ordinal_sha256": hashlib.sha256(b"3").hexdigest(),
            }
        ],
        "source_scan": {
            "fineweb": {"document_count": 4, "sha256": "a" * 64}
        },
        "training_authorized": False,
    }
    identity_payload = copy.deepcopy(report)
    identity_payload.pop("result_sha256")
    report["result_sha256"] = hashlib.sha256(
        canonical_json(identity_payload)
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_builds_deduplicated_quarantine(tmp_path: Path) -> None:
    report = build_quarantine(_review(tmp_path / "review.json"))
    assert report["candidate_count"] == 2
    assert report["quarantined_document_count"] == 1
    assert report["quarantined_documents"][0]["source_record_ordinal"] == 2
    assert report["training_authorized"] is False
    assert report["release_build_permitted"] is False


def test_rejects_mutated_review(tmp_path: Path) -> None:
    path = _review(tmp_path / "review.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["training_authorized"] = True
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        build_quarantine(path)


def test_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "quarantine.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_quarantine({}, output)


def test_writer_uses_lf_bytes(tmp_path: Path) -> None:
    output = tmp_path / "quarantine.json"
    write_quarantine({"value": "line"}, output)
    assert b"\r\n" not in output.read_bytes()


def test_real_independent_review_resolves_frozen_quarantine() -> None:
    report = build_quarantine(
        Path(
            "evaluation/results/"
            "vasu_140m_base_v2_semantic_contamination_independent_review_20260803.json"
        )
    )
    assert report["candidate_count"] == 2856
    assert report["quarantined_document_count"] == 2659
    assert report["quarantine_sha256"] == (
        "cfd31cf8ea68d27994b1d85caba163de1141a9ead1a8bc87c90ed39c7e67837b"
    )
    assert report["release_build_permitted"] is False
    assert report["training_authorized"] is False
