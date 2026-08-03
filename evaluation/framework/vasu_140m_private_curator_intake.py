"""Validate private VASU-140M held-out curator inputs without publishing text."""

from __future__ import annotations

import hashlib
import json
import operator
import os
import re
from collections.abc import Mapping
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file
from evaluation.framework.vasu_140m_base_v2_inventory import normalized_text_sha256


SCHEMA_ID = "vasu_140m_private_curator_intake_v1"
RECORD_SCHEMA_ID = "vasu_140m_sole_curator_draft_v1"
APPROVED_STATUS = "curator_approved_for_sealing"
COUNTS = {
    "factuality": 200,
    "arithmetic": 1_000,
    "repetition": 120,
    "robustness": 120,
    "manual_review": 60,
}
FILENAMES = {
    dimension: f"{dimension.replace('_', '-')}-heldout.jsonl"
    for dimension in COUNTS
}
BASE_FIELDS = {
    "schema_id",
    "status",
    "item_id",
    "dimension",
    "semantic_family_id",
    "parent_document_id",
    "provenance",
}
DIMENSION_FIELDS = {
    "factuality": {"prompt", "choices", "correct_choice_id"},
    "arithmetic": {"prompt", "answer_type", "expected_answer"},
    "repetition": {"prompt", "loop_ngram_size"},
    "robustness": {"baseline_prompt", "variant_prompt", "accepted_answers"},
    "manual_review": {"prompt", "rubric_dimensions"},
}
PROVENANCE_FIELDS = {
    "source_name",
    "source_url",
    "license_name",
    "license_url",
    "source_revision",
    "citation",
    "authored_by",
}


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact(value: Mapping[str, object], fields: set[str], label: str) -> None:
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _is_junction(path: Path) -> bool:
    function = getattr(os.path, "isjunction", None)
    return bool(function and function(path))


def _safe_private_directory(repository_root: Path, private_directory: Path) -> Path:
    repository = repository_root.resolve()
    unresolved = private_directory.absolute()
    if unresolved == repository or repository in unresolved.parents:
        raise ValueError("private curator directory must be outside repository")
    current = unresolved
    while current != current.parent:
        if current.exists() and (current.is_symlink() or _is_junction(current)):
            raise ValueError("private curator directory may not traverse links")
        current = current.parent
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise ValueError("private curator directory is missing")
    return resolved


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if path.is_symlink() or _is_junction(path) or not path.is_file():
        raise ValueError(f"private input is missing or linked: {path.name}")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            raise ValueError(f"{path.name} contains blank line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} line {line_number} must be an object")
        records.append(value)
    return records


def _validate_provenance(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be an object")
    _exact(value, PROVENANCE_FIELDS, "provenance")
    for field in PROVENANCE_FIELDS:
        _string(value[field], f"provenance.{field}")
    for field in ("source_url", "license_url"):
        if not str(value[field]).startswith("https://"):
            raise ValueError(f"provenance.{field} must use HTTPS")


def _validate_choices(record: Mapping[str, object]) -> None:
    choices = record["choices"]
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError("factuality choices must contain at least two options")
    identifiers: list[str] = []
    texts: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping) or set(choice) != {"choice_id", "text"}:
            raise ValueError("each factuality choice requires choice_id and text")
        identifiers.append(_string(choice["choice_id"], "choice_id"))
        texts.append(normalized_text_sha256(_string(choice["text"], "choice.text")))
    if len(identifiers) != len(set(identifiers)) or len(texts) != len(set(texts)):
        raise ValueError("factuality choices must be unique")
    if record["correct_choice_id"] not in identifiers:
        raise ValueError("correct_choice_id does not identify a choice")


_ARITHMETIC = re.compile(
    r"^Calculate exactly:\s*(-?\d+)\s*([+\-*/×÷])\s*(-?\d+)\s*=\s*\?$"
)
_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "×": operator.mul,
    "/": operator.floordiv,
    "÷": operator.floordiv,
}


def _validate_arithmetic(record: Mapping[str, object]) -> None:
    if record["answer_type"] != "integer":
        raise ValueError("arithmetic answer_type must be integer")
    expected_text = _string(record["expected_answer"], "expected_answer")
    try:
        expected = int(expected_text)
    except ValueError as error:
        raise ValueError("expected_answer must be an integer string") from error
    prompt = _string(record["prompt"], "prompt")
    match = _ARITHMETIC.fullmatch(prompt)
    if match is None:
        raise ValueError("arithmetic prompt must use the exact auditable template")
    left, symbol, right = int(match[1]), match[2], int(match[3])
    if symbol in {"/", "÷"} and (right == 0 or left % right != 0):
        raise ValueError("division must be nonzero and exact")
    if _OPERATORS[symbol](left, right) != expected:
        raise ValueError("arithmetic expected_answer is incorrect")


def _prompt_values(record: Mapping[str, object]) -> list[str]:
    if record["dimension"] == "robustness":
        return [str(record["baseline_prompt"]), str(record["variant_prompt"])]
    return [str(record["prompt"])]


def _validate_record(record: Mapping[str, object], dimension: str) -> None:
    _exact(record, BASE_FIELDS | DIMENSION_FIELDS[dimension], dimension)
    if record["schema_id"] != RECORD_SCHEMA_ID:
        raise ValueError("private curator record schema mismatch")
    if record["status"] != APPROVED_STATUS:
        raise ValueError("private curator record is not approved for sealing")
    if record["dimension"] != dimension:
        raise ValueError("record dimension does not match its file")
    for field in ("item_id", "semantic_family_id", "parent_document_id"):
        _string(record[field], field)
    _validate_provenance(record["provenance"])
    if dimension == "factuality":
        _string(record["prompt"], "prompt")
        _validate_choices(record)
    elif dimension == "arithmetic":
        _validate_arithmetic(record)
    elif dimension == "repetition":
        _string(record["prompt"], "prompt")
        if record["loop_ngram_size"] != 3:
            raise ValueError("repetition loop_ngram_size must equal 3")
    elif dimension == "robustness":
        baseline = _string(record["baseline_prompt"], "baseline_prompt")
        variant = _string(record["variant_prompt"], "variant_prompt")
        if normalized_text_sha256(baseline) == normalized_text_sha256(variant):
            raise ValueError("robustness prompts must differ after normalization")
        answers = record["accepted_answers"]
        if not isinstance(answers, list) or not answers:
            raise ValueError("robustness accepted_answers must be non-empty")
        for answer in answers:
            _string(answer, "accepted_answer")
    else:
        _string(record["prompt"], "prompt")
        rubrics = record["rubric_dimensions"]
        if rubrics != ["coherence", "factual_support", "degeneration"]:
            raise ValueError("manual-review rubric dimensions are not canonical")


def _development_prompt_hashes(repository_root: Path) -> set[str]:
    root = repository_root.resolve()
    suite = root / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
    hashes: set[str] = set()
    for dimension in COUNTS:
        payload = suite / dimension / "payload.jsonl"
        if not payload.is_file():
            raise ValueError("public development suite is incomplete")
        for record in _load_jsonl(payload):
            content = record.get("content")
            if not isinstance(content, Mapping):
                raise ValueError("public development payload is malformed")
            values = (
                [content["baseline_prompt"], content["variant_prompt"]]
                if dimension == "robustness"
                else [content["prompt"]]
            )
            hashes.update(normalized_text_sha256(str(value)) for value in values)
    return hashes


def report_identity(report: Mapping[str, object]) -> str:
    body = dict(report)
    body.pop("report_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def validate_private_curator_intake(
    repository_root: Path,
    private_directory: Path,
    *,
    expected_counts: Mapping[str, int] = COUNTS,
) -> dict[str, object]:
    """Return hash/count evidence without returning or writing prompt text."""

    root = repository_root.resolve()
    private = _safe_private_directory(root, private_directory)
    if set(expected_counts) != set(COUNTS):
        raise ValueError("expected_counts must cover exactly five dimensions")
    development_hashes = _development_prompt_hashes(root)
    item_ids: set[str] = set()
    families: set[str] = set()
    parents: set[str] = set()
    prompt_hashes: set[str] = set()
    files: dict[str, dict[str, object]] = {}
    for dimension, filename in FILENAMES.items():
        path = private / filename
        records = _load_jsonl(path)
        if len(records) != expected_counts[dimension]:
            raise ValueError(f"{dimension} record count mismatch")
        for record in records:
            _validate_record(record, dimension)
            for field, identities in (
                ("item_id", item_ids),
                ("semantic_family_id", families),
                ("parent_document_id", parents),
            ):
                identity = str(record[field])
                if identity in identities:
                    raise ValueError(f"duplicate {field} across held-out inputs")
                identities.add(identity)
            for prompt in _prompt_values(record):
                digest = normalized_text_sha256(prompt)
                if digest in prompt_hashes:
                    raise ValueError("duplicate prompt across held-out inputs")
                if digest in development_hashes:
                    raise ValueError("held-out prompt overlaps development prompt")
                prompt_hashes.add(digest)
        files[dimension] = {
            "filename": filename,
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
            "record_count": len(records),
        }
    report: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "files": files,
        "total_records": sum(int(value) for value in expected_counts.values()),
        "unique_item_ids": len(item_ids),
        "unique_semantic_families": len(families),
        "unique_parent_documents": len(parents),
        "unique_prompt_commitments": len(prompt_hashes),
        "development_overlap_count": 0,
        "plaintext_copied_to_repository": False,
        "private_key_opened": False,
        "held_out_opening_authorized": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }
    report["report_sha256"] = report_identity(report)
    return report
