"""Pre-split contamination scanning for future VASU-140M source documents."""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime

from evaluation.framework.vasu_140m_base_v2 import canonical_json
from evaluation.framework.vasu_140m_base_v2_inventory import (
    normalized_text_sha256,
    validate_contamination_record,
)


REPORT_SCHEMA_ID = "vasu_140m_base_source_contamination_scan_v1"
SEMANTIC_METHOD = "minhash_lsh_plus_human_v1"
FINDING_KINDS = frozenset({"prompt_exact", "answer_exact", "eight_word_fragment"})


def _exact(value: Mapping[str, object], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _timestamp(value: object, label: str) -> str:
    text = _string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.strip().split()))


def _span_hashes(text: str, widths: set[int]) -> dict[int, dict[str, list[int]]]:
    words = _normalized_text(text).split()
    result: dict[int, dict[str, list[int]]] = {}
    for width in sorted(widths):
        matches: dict[str, list[int]] = defaultdict(list)
        for index in range(max(0, len(words) - width + 1)):
            digest = normalized_text_sha256(" ".join(words[index : index + width]))
            matches[digest].append(index)
        result[width] = dict(matches)
    return result


def semantic_candidate_identity(candidate: Mapping[str, object]) -> str:
    body = dict(candidate)
    body.pop("candidate_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def validate_semantic_candidate(candidate: Mapping[str, object]) -> None:
    _exact(
        candidate,
        {
            "item_id",
            "document_id",
            "method",
            "similarity",
            "candidate_sha256",
        },
        "semantic candidate",
    )
    _string(candidate["item_id"], "semantic candidate.item_id")
    _string(candidate["document_id"], "semantic candidate.document_id")
    if candidate["method"] != SEMANTIC_METHOD:
        raise ValueError("semantic candidate method mismatch")
    similarity = candidate["similarity"]
    if (
        isinstance(similarity, bool)
        or not isinstance(similarity, (int, float))
        or not math.isfinite(float(similarity))
        or not 0.0 <= float(similarity) <= 1.0
    ):
        raise ValueError("semantic candidate similarity must be finite in [0, 1]")
    if _sha(candidate["candidate_sha256"], "candidate_sha256") != semantic_candidate_identity(candidate):
        raise ValueError("semantic candidate identity mismatch")


def validate_semantic_decision(decision: Mapping[str, object]) -> None:
    _exact(
        decision,
        {"candidate_sha256", "decision", "reviewed_by", "reviewed_at", "notes"},
        "semantic decision",
    )
    _sha(decision["candidate_sha256"], "semantic decision.candidate_sha256")
    if decision["decision"] not in {"clear", "reject"}:
        raise ValueError("semantic decision must be clear or reject")
    _string(decision["reviewed_by"], "semantic decision.reviewed_by")
    _timestamp(decision["reviewed_at"], "semantic decision.reviewed_at")
    _string(decision["notes"], "semantic decision.notes")


def report_identity(report: Mapping[str, object]) -> str:
    body = dict(report)
    body.pop("report_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def scan_source_documents(
    *,
    source_id: str,
    documents: Sequence[Mapping[str, object]],
    contamination_records: Sequence[Mapping[str, object]],
    semantic_candidates: Sequence[Mapping[str, object]],
    semantic_decisions: Sequence[Mapping[str, object]],
    semantic_search_evidence_sha256: str,
    fixture_only: bool,
) -> dict[str, object]:
    """Scan caller-provided pre-split documents without writing or acquiring data."""

    _string(source_id, "source_id")
    _sha(semantic_search_evidence_sha256, "semantic_search_evidence_sha256")
    if not isinstance(fixture_only, bool):
        raise ValueError("fixture_only must be boolean")
    if not documents or not contamination_records:
        raise ValueError("documents and contamination records must be non-empty")

    commitments: dict[str, Mapping[str, object]] = {}
    for record in contamination_records:
        validate_contamination_record(record)
        item_id = str(record["item_id"])
        if item_id in commitments:
            raise ValueError("contamination item IDs must be unique")
        commitments[item_id] = record

    candidate_map: dict[str, Mapping[str, object]] = {}
    pairs: set[tuple[str, str]] = set()
    for candidate in semantic_candidates:
        validate_semantic_candidate(candidate)
        digest = str(candidate["candidate_sha256"])
        pair = (str(candidate["item_id"]), str(candidate["document_id"]))
        if digest in candidate_map or pair in pairs:
            raise ValueError("semantic candidates must be unique")
        if pair[0] not in commitments:
            raise ValueError("semantic candidate references an unknown item")
        candidate_map[digest] = candidate
        pairs.add(pair)
    decision_map: dict[str, Mapping[str, object]] = {}
    for decision in semantic_decisions:
        validate_semantic_decision(decision)
        digest = str(decision["candidate_sha256"])
        if digest in decision_map:
            raise ValueError("semantic decisions must be unique")
        if digest not in candidate_map:
            raise ValueError("semantic decision references an unknown candidate")
        decision_map[digest] = decision
    if set(decision_map) != set(candidate_map):
        raise ValueError("every semantic candidate requires exactly one decision")

    exact_index: dict[tuple[int, str], list[tuple[str, str]]] = defaultdict(list)
    fragment_index: dict[str, list[str]] = defaultdict(list)
    widths: set[int] = set()
    for item_id, record in commitments.items():
        for field, kind in (
            ("prompt_exact_commitments", "prompt_exact"),
            ("answer_exact_commitments", "answer_exact"),
        ):
            for raw in record[field]:
                commitment = _mapping(raw, f"{field} commitment")
                width = int(commitment["word_count"])
                widths.add(width)
                exact_index[(width, str(commitment["sha256"]))].append((item_id, kind))
        for digest in record["ngram_sha256s"]:
            fragment_index[str(digest)].append(item_id)
        widths.add(int(record["ngram_words"]))

    document_ids: set[str] = set()
    document_evidence: list[dict[str, object]] = []
    findings: list[dict[str, object]] = []
    for raw_document in documents:
        document = _mapping(raw_document, "source document")
        _exact(document, {"document_id", "source_id", "text"}, "source document")
        document_id = _string(document["document_id"], "document_id")
        if document_id in document_ids:
            raise ValueError("source document IDs must be unique")
        document_ids.add(document_id)
        if document["source_id"] != source_id:
            raise ValueError("source document source identity mismatch")
        text = _string(document["text"], "source document.text")
        normalized = _normalized_text(text)
        span_hashes = _span_hashes(normalized, widths)
        document_evidence.append(
            {
                "document_id": document_id,
                "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "normalized_sha256": normalized_text_sha256(text),
                "word_count": len(normalized.split()),
            }
        )
        observed: set[tuple[str, str, str, int, int]] = set()
        for (width, digest), references in exact_index.items():
            for start in span_hashes[width].get(digest, []):
                for item_id, kind in references:
                    observed.add((item_id, kind, digest, width, start))
        for digest, item_ids in fragment_index.items():
            for width in {
                int(commitments[item_id]["ngram_words"]) for item_id in item_ids
            }:
                for start in span_hashes[width].get(digest, []):
                    for item_id in item_ids:
                        if int(commitments[item_id]["ngram_words"]) == width:
                            observed.add(
                                (item_id, "eight_word_fragment", digest, width, start)
                            )
        findings.extend(
            {
                "document_id": document_id,
                "item_id": item_id,
                "kind": kind,
                "commitment_sha256": digest,
                "word_count": width,
                "span_start": start,
            }
            for item_id, kind, digest, width, start in sorted(observed)
        )
    unknown_documents = {
        str(candidate["document_id"])
        for candidate in semantic_candidates
        if str(candidate["document_id"]) not in document_ids
    }
    if unknown_documents:
        raise ValueError("semantic candidate references an unknown document")

    exact_rejected = {str(finding["document_id"]) for finding in findings}
    semantic_rejected = {
        str(candidate_map[digest]["document_id"])
        for digest, decision in decision_map.items()
        if decision["decision"] == "reject"
    }
    rejected = exact_rejected | semantic_rejected
    outcomes = [
        {
            "document_id": item["document_id"],
            "decision": "reject" if item["document_id"] in rejected else "eligible",
        }
        for item in document_evidence
    ]
    report: dict[str, object] = {
        "schema_id": REPORT_SCHEMA_ID,
        "source_id": source_id,
        "document_evidence": document_evidence,
        "contamination_inventory_sha256": hashlib.sha256(
            canonical_json(list(contamination_records))
        ).hexdigest(),
        "semantic_search": {
            "method": SEMANTIC_METHOD,
            "evidence_sha256": semantic_search_evidence_sha256,
            "candidates_sha256": hashlib.sha256(
                canonical_json(list(semantic_candidates))
            ).hexdigest(),
            "decisions_sha256": hashlib.sha256(
                canonical_json(list(semantic_decisions))
            ).hexdigest(),
            "candidate_count": len(candidate_map),
            "decision_count": len(decision_map),
            "complete": True,
        },
        "exact_findings": findings,
        "semantic_rejections": sorted(semantic_rejected),
        "document_outcomes": outcomes,
        "counts": {
            "documents": len(document_evidence),
            "exact_findings": len(findings),
            "semantic_candidates": len(candidate_map),
            "semantic_rejections": len(semantic_rejected),
            "eligible_documents": len(document_evidence) - len(rejected),
            "rejected_documents": len(rejected),
        },
        "scan_before_split": True,
        "scan_complete": True,
        "fixture_only": fixture_only,
        "source_admission_approved": False,
        "release_build_permitted": False,
        "training_authorized": False,
    }
    report["report_sha256"] = report_identity(report)
    validate_scan_report(report)
    return report


def validate_scan_report(report: Mapping[str, object]) -> None:
    """Validate immutable scan evidence without claiming source admission."""

    _exact(
        report,
        {
            "schema_id",
            "source_id",
            "document_evidence",
            "contamination_inventory_sha256",
            "semantic_search",
            "exact_findings",
            "semantic_rejections",
            "document_outcomes",
            "counts",
            "scan_before_split",
            "scan_complete",
            "fixture_only",
            "source_admission_approved",
            "release_build_permitted",
            "training_authorized",
            "report_sha256",
        },
        "scan report",
    )
    if report["schema_id"] != REPORT_SCHEMA_ID:
        raise ValueError("scan report schema mismatch")
    _string(report["source_id"], "source_id")
    _sha(report["contamination_inventory_sha256"], "contamination_inventory_sha256")
    documents = report["document_evidence"]
    if not isinstance(documents, list) or not documents:
        raise ValueError("document_evidence must be a non-empty list")
    document_ids: list[str] = []
    for raw in documents:
        document = _mapping(raw, "document evidence")
        _exact(
            document,
            {"document_id", "raw_sha256", "normalized_sha256", "word_count"},
            "document evidence",
        )
        document_ids.append(_string(document["document_id"], "document_id"))
        _sha(document["raw_sha256"], "document raw SHA-256")
        _sha(document["normalized_sha256"], "document normalized SHA-256")
        _nonnegative_int(document["word_count"], "document word_count")
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document evidence IDs must be unique")
    semantic = _mapping(report["semantic_search"], "semantic_search")
    _exact(
        semantic,
        {
            "method",
            "evidence_sha256",
            "candidates_sha256",
            "decisions_sha256",
            "candidate_count",
            "decision_count",
            "complete",
        },
        "semantic_search",
    )
    if semantic["method"] != SEMANTIC_METHOD or semantic["complete"] is not True:
        raise ValueError("semantic search evidence is incomplete")
    _sha(semantic["evidence_sha256"], "semantic_search.evidence_sha256")
    _sha(semantic["candidates_sha256"], "semantic_search.candidates_sha256")
    _sha(semantic["decisions_sha256"], "semantic_search.decisions_sha256")
    candidate_count = _nonnegative_int(
        semantic["candidate_count"], "semantic_search.candidate_count"
    )
    decision_count = _nonnegative_int(
        semantic["decision_count"], "semantic_search.decision_count"
    )
    if candidate_count != decision_count:
        raise ValueError("semantic candidate decisions are incomplete")
    findings = report["exact_findings"]
    if not isinstance(findings, list):
        raise ValueError("exact_findings must be a list")
    for finding in findings:
        value = _mapping(finding, "exact finding")
        _exact(
            value,
            {"document_id", "item_id", "kind", "commitment_sha256", "word_count", "span_start"},
            "exact finding",
        )
        if value["kind"] not in FINDING_KINDS:
            raise ValueError("exact finding kind mismatch")
        if value["document_id"] not in document_ids:
            raise ValueError("exact finding references an unknown document")
        _string(value["item_id"], "finding item_id")
        _sha(value["commitment_sha256"], "finding commitment")
        if _nonnegative_int(value["word_count"], "finding word_count") == 0:
            raise ValueError("finding word_count must be positive")
        _nonnegative_int(value["span_start"], "finding span_start")
    semantic_rejections = report["semantic_rejections"]
    if (
        not isinstance(semantic_rejections, list)
        or any(item not in document_ids for item in semantic_rejections)
        or len(semantic_rejections) != len(set(semantic_rejections))
    ):
        raise ValueError("semantic_rejections must be unique known documents")
    outcomes = report["document_outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != len(document_ids):
        raise ValueError("document_outcomes must cover every document")
    outcome_map: dict[str, str] = {}
    for raw in outcomes:
        outcome = _mapping(raw, "document outcome")
        _exact(outcome, {"document_id", "decision"}, "document outcome")
        document_id = _string(outcome["document_id"], "outcome document_id")
        if document_id in outcome_map or document_id not in document_ids:
            raise ValueError("document outcomes contain unknown or duplicate IDs")
        if outcome["decision"] not in {"eligible", "reject"}:
            raise ValueError("document outcome decision mismatch")
        outcome_map[document_id] = str(outcome["decision"])
    finding_rejections = {str(item["document_id"]) for item in findings}
    expected_rejections = finding_rejections | set(semantic_rejections)
    observed_rejections = {
        document_id
        for document_id, decision in outcome_map.items()
        if decision == "reject"
    }
    if expected_rejections != observed_rejections:
        raise ValueError("document outcomes do not match contamination findings")
    counts = _mapping(report["counts"], "counts")
    _exact(
        counts,
        {
            "documents",
            "exact_findings",
            "semantic_candidates",
            "semantic_rejections",
            "eligible_documents",
            "rejected_documents",
        },
        "counts",
    )
    expected_counts = {
        "documents": len(document_ids),
        "exact_findings": len(findings),
        "semantic_candidates": candidate_count,
        "semantic_rejections": len(semantic_rejections),
        "eligible_documents": len(document_ids) - len(expected_rejections),
        "rejected_documents": len(expected_rejections),
    }
    for name, expected in expected_counts.items():
        _nonnegative_int(counts[name], f"counts.{name}")
        if counts[name] != expected:
            raise ValueError("scan report counts do not match evidence")
    for flag in ("scan_before_split", "scan_complete"):
        if report[flag] is not True:
            raise ValueError(f"{flag} must be true")
    if not isinstance(report["fixture_only"], bool):
        raise ValueError("fixture_only must be boolean")
    for flag in (
        "source_admission_approved",
        "release_build_permitted",
        "training_authorized",
    ):
        if report[flag] is not False:
            raise ValueError(f"{flag} must be false")
    if _sha(report["report_sha256"], "report_sha256") != report_identity(report):
        raise ValueError("scan report identity mismatch")
