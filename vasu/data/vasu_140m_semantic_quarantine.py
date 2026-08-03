"""Build a source-document quarantine from independent opaque review evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file

SCHEMA_ID = "vasu_140m_base_v2_semantic_contamination_independent_review_v1"
OUTPUT_SCHEMA_ID = "vasu_140m_source_semantic_quarantine_v1"
QUARANTINE_DECISIONS = {
    "quarantine_exact_prompt_or_answer_overlap",
    "quarantine_for_source_admission_review",
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _embedded_identity(report: Mapping[str, object]) -> str:
    payload = dict(report)
    expected = payload.pop("result_sha256", None)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("review result_sha256 is missing")
    observed = hashlib.sha256(canonical_json(payload)).hexdigest()
    if observed != expected:
        raise ValueError("review result identity mismatch")
    return observed


def _ordinal_from_commitment(value: object, ordinals: Mapping[str, int]) -> int:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("source ordinal commitment is invalid")
    try:
        return ordinals[value]
    except KeyError as error:
        raise ValueError("source ordinal commitment cannot be resolved") from error


def build_quarantine(
    review_path: Path,
    *,
    source_id: str = "fineweb_edu_extension_2025_26",
) -> dict[str, object]:
    """Return a deterministic quarantine without opening held-out payloads."""

    report = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_id") != SCHEMA_ID:
        raise ValueError("unexpected independent review schema")
    result_sha256 = _embedded_identity(report)
    if report.get("private_key_opened") is not False:
        raise ValueError("review must prove that the private key stayed closed")
    if report.get("held_out_content_exposed") is not False:
        raise ValueError("review must prove that held-out content stayed private")
    if report.get("training_authorized") is not False:
        raise ValueError("review must remain non-authorizing")

    source_scan = _mapping(report.get("source_scan"), "source_scan")
    fineweb = _mapping(source_scan.get("fineweb"), "source_scan.fineweb")
    document_count = fineweb.get("documents_scanned", fineweb.get("document_count"))
    source_sha256 = fineweb.get("source_sha256", fineweb.get("sha256"))
    if not isinstance(document_count, int) or document_count <= 0:
        raise ValueError("FineWeb document count is invalid")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("FineWeb source identity is invalid")

    ordinal_commitments = {
        hashlib.sha256(str(ordinal).encode("ascii")).hexdigest(): ordinal
        for ordinal in range(document_count)
    }
    candidates = [
        *_list(report.get("exact_candidates"), "exact_candidates"),
        *_list(report.get("semantic_candidates"), "semantic_candidates"),
    ]
    resolved: dict[int, set[str]] = {}
    candidate_count = 0
    for raw in candidates:
        candidate = _mapping(raw, "candidate")
        if candidate.get("decision") not in QUARANTINE_DECISIONS:
            continue
        if candidate.get("source") != "fineweb":
            raise ValueError("unexpected quarantined source")
        ordinal = _ordinal_from_commitment(
            candidate.get("source_record_ordinal_sha256"), ordinal_commitments
        )
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate ID is invalid")
        resolved.setdefault(ordinal, set()).add(candidate_id)
        candidate_count += 1

    records = [
        {
            "candidate_ids": sorted(candidate_ids),
            "source_record_ordinal": ordinal,
            "source_record_ordinal_sha256": hashlib.sha256(
                str(ordinal).encode("ascii")
            ).hexdigest(),
        }
        for ordinal, candidate_ids in sorted(resolved.items())
    ]
    output: dict[str, object] = {
        "candidate_count": candidate_count,
        "decision": "quarantine_source_documents",
        "evaluation_run_authorized": False,
        "independent_review": {
            "path": review_path.as_posix(),
            "result_sha256": result_sha256,
            "sha256": sha256_file(review_path),
        },
        "quarantined_document_count": len(records),
        "quarantined_documents": records,
        "release_build_permitted": False,
        "schema_id": OUTPUT_SCHEMA_ID,
        "source": {
            "document_count": document_count,
            "sha256": source_sha256,
            "source_id": source_id,
        },
        "training_authorized": False,
    }
    output["quarantine_sha256"] = hashlib.sha256(canonical_json(output)).hexdigest()
    return output


def write_quarantine(report: Mapping[str, object], output_path: Path) -> None:
    """Write a new quarantine atomically and refuse replacement."""

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"staging path already exists: {temporary}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
