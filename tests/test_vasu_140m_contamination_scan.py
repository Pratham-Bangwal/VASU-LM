from __future__ import annotations

import copy
import hashlib

import pytest

from evaluation.framework.vasu_140m_base_v2 import canonical_json
from evaluation.framework.vasu_140m_base_v2_inventory import (
    CONTAMINATION_RECORD_SCHEMA_ID,
    normalized_text_sha256,
)
from vasu.data.vasu_140m_contamination_scan import (
    SEMANTIC_METHOD,
    report_identity,
    scan_source_documents,
    semantic_candidate_identity,
    validate_scan_report,
)


SEMANTIC_EVIDENCE = "e" * 64


def _record() -> dict[str, object]:
    prompt = "The hidden evaluation prompt asks for a careful factual continuation"
    answer = "Mars"
    fragment = "hidden evaluation prompt asks for a careful factual"
    return {
        "schema_id": CONTAMINATION_RECORD_SCHEMA_ID,
        "item_id": "factuality-001",
        "parent_document_id": "benchmark-parent-001",
        "prompt_exact_commitments": [
            {"sha256": normalized_text_sha256(prompt), "word_count": 10}
        ],
        "answer_exact_commitments": [
            {"sha256": normalized_text_sha256(answer), "word_count": 1}
        ],
        "ngram_words": 8,
        "ngram_sha256s": [normalized_text_sha256(fragment)],
        "semantic_fingerprint": {
            "method": "fixture-semantic-v1",
            "value": "f" * 64,
        },
    }


def _candidate(document_id: str = "doc-002") -> dict[str, object]:
    candidate: dict[str, object] = {
        "item_id": "factuality-001",
        "document_id": document_id,
        "method": SEMANTIC_METHOD,
        "similarity": 0.81,
    }
    candidate["candidate_sha256"] = semantic_candidate_identity(candidate)
    return candidate


def _decision(candidate: dict[str, object], decision: str = "clear") -> dict[str, object]:
    return {
        "candidate_sha256": candidate["candidate_sha256"],
        "decision": decision,
        "reviewed_by": "Independent fixture reviewer",
        "reviewed_at": "2026-08-01T13:00:00+05:30",
        "notes": "Synthetic qualification decision.",
    }


def _documents() -> list[dict[str, str]]:
    return [
        {
            "document_id": "doc-001",
            "source_id": "fixture-source",
            "text": (
                "Introductory words. The hidden evaluation prompt asks for a careful "
                "factual continuation before unrelated closing words."
            ),
        },
        {
            "document_id": "doc-002",
            "source_id": "fixture-source",
            "text": "A separate article says mars is visible in the evening sky.",
        },
        {
            "document_id": "doc-003",
            "source_id": "fixture-source",
            "text": "This unrelated document discusses rainfall and river systems.",
        },
    ]


def _scan(*, semantic_decision: str = "clear") -> dict[str, object]:
    candidate = _candidate()
    return scan_source_documents(
        source_id="fixture-source",
        documents=_documents(),
        contamination_records=[_record()],
        semantic_candidates=[candidate],
        semantic_decisions=[_decision(candidate, semantic_decision)],
        semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
        fixture_only=True,
    )


def test_detects_embedded_prompt_short_answer_and_fragment() -> None:
    report = _scan()
    validate_scan_report(report)
    candidate = _candidate()
    assert report["semantic_search"]["candidates_sha256"] == hashlib.sha256(
        canonical_json([candidate])
    ).hexdigest()
    assert report["semantic_search"]["decisions_sha256"] == hashlib.sha256(
        canonical_json([_decision(candidate)])
    ).hexdigest()
    findings = {
        (item["document_id"], item["kind"]) for item in report["exact_findings"]
    }
    assert ("doc-001", "prompt_exact") in findings
    assert ("doc-001", "eight_word_fragment") in findings
    assert ("doc-002", "answer_exact") in findings
    assert report["counts"]["rejected_documents"] == 2
    assert report["counts"]["eligible_documents"] == 1
    assert report["scan_before_split"] is True
    assert report["source_admission_approved"] is False
    assert report["release_build_permitted"] is False
    assert report["training_authorized"] is False


def test_semantic_reject_blocks_otherwise_clean_document() -> None:
    documents = _documents()
    documents[1]["text"] = "A clean document with no committed answer token."
    candidate = _candidate()
    report = scan_source_documents(
        source_id="fixture-source",
        documents=documents,
        contamination_records=[_record()],
        semantic_candidates=[candidate],
        semantic_decisions=[_decision(candidate, "reject")],
        semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
        fixture_only=True,
    )
    assert report["semantic_rejections"] == ["doc-002"]
    assert {item["document_id"] for item in report["document_outcomes"] if item["decision"] == "reject"} == {
        "doc-001",
        "doc-002",
    }


def test_missing_or_unknown_semantic_decision_fails_closed() -> None:
    candidate = _candidate()
    with pytest.raises(ValueError, match="every semantic candidate"):
        scan_source_documents(
            source_id="fixture-source",
            documents=_documents(),
            contamination_records=[_record()],
            semantic_candidates=[candidate],
            semantic_decisions=[],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )
    unknown = _decision(candidate)
    unknown["candidate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unknown candidate"):
        scan_source_documents(
            source_id="fixture-source",
            documents=_documents(),
            contamination_records=[_record()],
            semantic_candidates=[candidate],
            semantic_decisions=[unknown],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )


def test_candidate_references_must_resolve() -> None:
    unknown_item = _candidate()
    unknown_item["item_id"] = "unknown-item"
    unknown_item["candidate_sha256"] = semantic_candidate_identity(unknown_item)
    with pytest.raises(ValueError, match="unknown item"):
        scan_source_documents(
            source_id="fixture-source",
            documents=_documents(),
            contamination_records=[_record()],
            semantic_candidates=[unknown_item],
            semantic_decisions=[_decision(unknown_item)],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )
    unknown_document = _candidate("missing-document")
    with pytest.raises(ValueError, match="unknown document"):
        scan_source_documents(
            source_id="fixture-source",
            documents=_documents(),
            contamination_records=[_record()],
            semantic_candidates=[unknown_document],
            semantic_decisions=[_decision(unknown_document)],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )


def test_source_and_document_id_mismatch_fail_closed() -> None:
    documents = _documents()
    documents[0]["source_id"] = "substituted-source"
    with pytest.raises(ValueError, match="source identity"):
        scan_source_documents(
            source_id="fixture-source",
            documents=documents,
            contamination_records=[_record()],
            semantic_candidates=[],
            semantic_decisions=[],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )
    documents = _documents()
    documents[1]["document_id"] = documents[0]["document_id"]
    with pytest.raises(ValueError, match="document IDs"):
        scan_source_documents(
            source_id="fixture-source",
            documents=documents,
            contamination_records=[_record()],
            semantic_candidates=[],
            semantic_decisions=[],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )


def test_candidate_identity_and_similarity_are_strict() -> None:
    candidate = _candidate()
    candidate["similarity"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        scan_source_documents(
            source_id="fixture-source",
            documents=_documents(),
            contamination_records=[_record()],
            semantic_candidates=[candidate],
            semantic_decisions=[],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )
    candidate = _candidate()
    candidate["similarity"] = 0.82
    with pytest.raises(ValueError, match="identity mismatch"):
        scan_source_documents(
            source_id="fixture-source",
            documents=_documents(),
            contamination_records=[_record()],
            semantic_candidates=[candidate],
            semantic_decisions=[],
            semantic_search_evidence_sha256=SEMANTIC_EVIDENCE,
            fixture_only=True,
        )


def test_report_identity_rejects_mutation() -> None:
    report = _scan()
    changed = copy.deepcopy(report)
    changed["training_authorized"] = True
    changed["report_sha256"] = report_identity(changed)
    with pytest.raises(ValueError, match="training_authorized"):
        validate_scan_report(changed)
    changed = copy.deepcopy(report)
    changed["counts"]["eligible_documents"] += 1
    changed["report_sha256"] = report_identity(changed)
    with pytest.raises(ValueError, match="counts"):
        validate_scan_report(changed)
    changed = copy.deepcopy(report)
    changed["semantic_search"]["candidate_count"] = True
    changed["semantic_search"]["decision_count"] = True
    changed["report_sha256"] = report_identity(changed)
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_scan_report(changed)


def test_inventory_identity_is_order_sensitive() -> None:
    record = _record()
    report = _scan()
    expected = hashlib.sha256(canonical_json([record])).hexdigest()
    assert report["contamination_inventory_sha256"] == expected
