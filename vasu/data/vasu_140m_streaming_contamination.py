"""Streaming hash-only contamination scans for acquired VASU-140M sources."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from evaluation.framework.vasu_140m_base_v2 import canonical_json
from evaluation.framework.vasu_140m_base_v2_inventory import (
    normalized_text_sha256,
    validate_contamination_record,
)


SCHEMA_ID = "vasu_140m_streaming_exact_contamination_scan_v2"
MAX_FINDING_SAMPLE = 100


def report_identity(report: Mapping[str, object]) -> str:
    body = dict(report)
    body.pop("report_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def build_commitment_index(
    records: Sequence[Mapping[str, object]],
) -> dict[int, dict[str, tuple[tuple[str, str, bool], ...]]]:
    """Index non-redundant short exact spans and all eight-word fragments."""

    index: dict[int, dict[str, set[tuple[str, str, bool]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for record in records:
        validate_contamination_record(record)
        item_id = str(record["item_id"])
        width = int(record["ngram_words"])
        for digest in record["ngram_sha256s"]:
            index[width][str(digest)].add((item_id, "fragment", True))
        for field, kind, blocking in (
            ("prompt_exact_commitments", "prompt_exact", True),
            ("answer_exact_commitments", "answer_exact", False),
        ):
            for commitment in record[field]:
                exact_width = int(commitment["word_count"])
                if exact_width < width:
                    index[exact_width][str(commitment["sha256"])].add(
                        (item_id, kind, blocking)
                    )
    return {
        width: {digest: tuple(sorted(ids)) for digest, ids in values.items()}
        for width, values in index.items()
    }


def scan_documents(
    *,
    source_id: str,
    documents: Iterable[Mapping[str, str]],
    excluded_parent_ids: set[str],
    contamination_records: Sequence[Mapping[str, object]],
    source_artifact_sha256: str,
) -> dict[str, object]:
    index = build_commitment_index(contamination_records)
    findings: list[dict[str, object]] = []
    findings_digest = hashlib.sha256()
    matched_document_count = 0
    document_count = 0
    excluded_count = 0
    scanned_count = 0
    rejected_parents: set[str] = set()
    audit_only_parents: set[str] = set()
    seen: set[str] = set()
    for document in documents:
        document_id = str(document["document_id"])
        parent_id = str(document["parent_document_id"])
        if document_id in seen:
            raise ValueError("source document IDs must be unique")
        seen.add(document_id)
        document_count += 1
        if parent_id in excluded_parent_ids:
            excluded_count += 1
            continue
        scanned_count += 1
        words = unicodedata.normalize(
            "NFC", " ".join(str(document["text"]).strip().split())
        ).split()
        observed: set[tuple[str, str, bool, int, int]] = set()
        for width, commitments in index.items():
            for start in range(max(0, len(words) - width + 1)):
                digest = normalized_text_sha256(" ".join(words[start : start + width]))
                for item_id, kind, blocking in commitments.get(digest, ()):
                    observed.add((item_id, kind, blocking, width, start))
        if observed:
            if any(match[2] for match in observed):
                rejected_parents.add(parent_id)
            else:
                audit_only_parents.add(parent_id)
            finding = {
                    "document_id": document_id,
                    "parent_document_id": parent_id,
                    "document_sha256": hashlib.sha256(
                        str(document["text"]).encode("utf-8")
                    ).hexdigest(),
                    "matches": [
                        {
                            "item_id": item_id,
                            "kind": kind,
                            "disposition": "reject" if blocking else "audit_only",
                            "word_count": width,
                            "span_start": start,
                        }
                        for item_id, kind, blocking, width, start in sorted(observed)
                    ],
                }
            findings_digest.update(canonical_json(finding))
            matched_document_count += 1
            if len(findings) < MAX_FINDING_SAMPLE:
                findings.append(finding)
    report: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "source_id": source_id,
        "source_artifact_sha256": source_artifact_sha256,
        "contamination_inventory_sha256": hashlib.sha256(
            canonical_json(list(contamination_records))
        ).hexdigest(),
        "excluded_parent_ids_sha256": hashlib.sha256(
            canonical_json(sorted(excluded_parent_ids))
        ).hexdigest(),
        "counts": {
            "documents": document_count,
            "excluded_documents": excluded_count,
            "scanned_documents": scanned_count,
            "matched_documents": matched_document_count,
            "rejected_parent_documents": len(rejected_parents),
            "audit_only_parent_documents": len(audit_only_parents - rejected_parents),
        },
        "findings": findings,
        "findings_sample_limit": MAX_FINDING_SAMPLE,
        "findings_truncated": matched_document_count > len(findings),
        "findings_sha256": findings_digest.hexdigest(),
        "exact_and_fragment_scan_complete": True,
        "semantic_scan_complete": False,
        "source_admission_approved": False,
        "release_build_permitted": False,
        "training_authorized": False,
    }
    report["report_sha256"] = report_identity(report)
    validate_streaming_report(report)
    return report


def validate_streaming_report(report: Mapping[str, object]) -> None:
    required = {
        "schema_id", "source_id", "source_artifact_sha256",
        "contamination_inventory_sha256", "excluded_parent_ids_sha256",
        "counts", "findings", "findings_sample_limit", "findings_truncated",
        "findings_sha256", "exact_and_fragment_scan_complete",
        "semantic_scan_complete", "source_admission_approved",
        "release_build_permitted", "training_authorized", "report_sha256",
    }
    if set(report) != required or report["schema_id"] != SCHEMA_ID:
        raise ValueError("streaming scan schema mismatch")
    for field in (
        "source_artifact_sha256", "contamination_inventory_sha256",
        "excluded_parent_ids_sha256", "findings_sha256", "report_sha256",
    ):
        value = report[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field} must be a SHA-256")
    if report["exact_and_fragment_scan_complete"] is not True:
        raise ValueError("exact and fragment scan must be complete")
    for field in (
        "semantic_scan_complete", "source_admission_approved",
        "release_build_permitted", "training_authorized",
    ):
        if report[field] is not False:
            raise ValueError(f"{field} must remain false")
    counts = report["counts"]
    if not isinstance(counts, Mapping) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counts.values()
    ):
        raise ValueError("scan counts must be non-negative integers")
    if counts["documents"] != counts["excluded_documents"] + counts["scanned_documents"]:
        raise ValueError("scan document counts mismatch")
    findings = report["findings"]
    if not isinstance(findings, list) or len(findings) > report["findings_sample_limit"]:
        raise ValueError("scan finding sample is invalid")
    expected_truncation = counts["matched_documents"] > len(findings)
    if report["findings_truncated"] is not expected_truncation:
        raise ValueError("scan finding truncation flag mismatch")
    if report["report_sha256"] != report_identity(report):
        raise ValueError("streaming scan identity mismatch")
