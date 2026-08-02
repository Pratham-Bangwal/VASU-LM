from __future__ import annotations

import copy

import pytest

from evaluation.framework.vasu_140m_base_v2_inventory import (
    CONTAMINATION_RECORD_SCHEMA_ID,
    normalized_text_sha256,
)
from vasu.data.vasu_140m_streaming_contamination import (
    report_identity,
    scan_documents,
    validate_streaming_report,
)


def _record() -> dict[str, object]:
    prompt = "one two three four five six seven eight nine"
    return {
        "schema_id": CONTAMINATION_RECORD_SCHEMA_ID,
        "item_id": "item-1",
        "parent_document_id": "eval-parent",
        "prompt_exact_commitments": [{"sha256": normalized_text_sha256(prompt), "word_count": 9}],
        "answer_exact_commitments": [{"sha256": normalized_text_sha256("Mars"), "word_count": 1}],
        "ngram_words": 8,
        "ngram_sha256s": [normalized_text_sha256("one two three four five six seven eight")],
        "semantic_fingerprint": {"method": "fixture", "value": "f" * 64},
    }


def test_streaming_scan_excludes_reserved_and_detects_hash_spans() -> None:
    documents = [
        {"document_id": "doc-a", "parent_document_id": "reserved", "text": "Mars"},
        {"document_id": "doc-b", "parent_document_id": "parent-b", "text": "Visible mars tonight"},
        {"document_id": "doc-c", "parent_document_id": "parent-c", "text": "one two three four five six seven eight extra"},
    ]
    report = scan_documents(
        source_id="source", documents=documents, excluded_parent_ids={"reserved"},
        contamination_records=[_record()], source_artifact_sha256="a" * 64,
    )
    assert report["counts"] == {
        "documents": 3, "excluded_documents": 1, "scanned_documents": 2,
        "matched_documents": 2, "rejected_parent_documents": 2,
    }
    assert report["semantic_scan_complete"] is False
    assert report["training_authorized"] is False


def test_streaming_report_mutation_fails() -> None:
    report = scan_documents(
        source_id="source",
        documents=[{"document_id": "doc", "parent_document_id": "parent", "text": "clear text"}],
        excluded_parent_ids=set(), contamination_records=[_record()],
        source_artifact_sha256="a" * 64,
    )
    changed = copy.deepcopy(report)
    changed["training_authorized"] = True
    changed["report_sha256"] = report_identity(changed)
    with pytest.raises(ValueError, match="must remain false"):
        validate_streaming_report(changed)
