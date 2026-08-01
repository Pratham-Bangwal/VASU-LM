"""Synthetic pre-split contamination scan; no source acquisition or release."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_base_v2_inventory import (  # noqa: E402
    CONTAMINATION_RECORD_SCHEMA_ID,
    normalized_text_sha256,
)
from vasu.data.vasu_140m_contamination_scan import (  # noqa: E402
    SEMANTIC_METHOD,
    scan_source_documents,
    semantic_candidate_identity,
)


def build_report() -> dict[str, object]:
    prompt = "The hidden evaluation prompt asks for a careful factual continuation"
    answer = "Mars"
    fragment = "hidden evaluation prompt asks for a careful factual"
    contamination = {
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
            "method": "synthetic-semantic-v1",
            "value": "f" * 64,
        },
    }
    candidate: dict[str, object] = {
        "item_id": "factuality-001",
        "document_id": "doc-003",
        "method": SEMANTIC_METHOD,
        "similarity": 0.79,
    }
    candidate["candidate_sha256"] = semantic_candidate_identity(candidate)
    decision = {
        "candidate_sha256": candidate["candidate_sha256"],
        "decision": "clear",
        "reviewed_by": "Independent synthetic fixture reviewer",
        "reviewed_at": "2026-08-01T13:00:00+05:30",
        "notes": "Synthetic candidate is intentionally unrelated.",
    }
    return scan_source_documents(
        source_id="fixture-source",
        documents=[
            {
                "document_id": "doc-001",
                "source_id": "fixture-source",
                "text": (
                    "Intro. The hidden evaluation prompt asks for a careful factual "
                    "continuation before unrelated closing words."
                ),
            },
            {
                "document_id": "doc-002",
                "source_id": "fixture-source",
                "text": "An astronomy note says mars is visible in the evening sky.",
            },
            {
                "document_id": "doc-003",
                "source_id": "fixture-source",
                "text": "An unrelated article discusses rainfall and river systems.",
            },
        ],
        contamination_records=[contamination],
        semantic_candidates=[candidate],
        semantic_decisions=[decision],
        semantic_search_evidence_sha256="e" * 64,
        fixture_only=True,
    )


def main() -> int:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["scan_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
