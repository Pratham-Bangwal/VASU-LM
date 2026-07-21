"""Versioned validation and release utilities for reviewed instruction data."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np

from vasu.data.alpaca_masked_v2 import build_packed_records
from vasu.tokenizer.tokenizer import VASUTokenizer


SCHEMA_VERSION = "vasu_instruction_quality_v1"
REVIEW_SCHEMA_VERSION = "vasu_instruction_quality_review_v1"
RELEASE_VERSION = "vasu_instruction_quality_release_v1"
CAPABILITIES = (
    "short_factual_qa",
    "beginner_explanation",
    "exact_format_following",
    "rewriting_transformation",
    "lists_structured_output",
    "json_schema_output",
    "uncertainty_honest_fallback",
)
REVIEW_STATUSES = (
    "unreviewed",
    "approved",
    "minor_issue",
    "rejected",
    "needs_fact_check",
    "needs_rewrite",
)
FINAL_REVIEW_STATUSES = frozenset({"approved", "rejected"})
FORMAT_TYPES = (
    "none",
    "exact_bullets",
    "exact_sentences",
    "maximum_words",
    "one_word",
    "json_object",
    "ordered_list",
    "unordered_list",
    "no_list",
    "required_headings",
    "required_fields",
    "plain_text_only",
    "exact_numbered_items",
    "exact_labeled_fields",
    "exact_words",
)
DIFFICULTIES = ("easy", "medium", "hard")
SOURCE_TYPES = ("human_written", "curated", "synthetic_demo")
TRANSITIONS = {
    "unreviewed": set(REVIEW_STATUSES) - {"unreviewed"},
    "minor_issue": {"approved", "rejected", "needs_rewrite"},
    "needs_fact_check": {"approved", "rejected", "needs_rewrite"},
    "needs_rewrite": {"approved", "rejected", "needs_fact_check"},
    "approved": {"needs_rewrite", "needs_fact_check", "rejected"},
    "rejected": {"needs_rewrite", "needs_fact_check"},
}
TEMPLATE_MARKERS = ("User:", "Assistant:", "### Instruction:", "### Response:")
PLACEHOLDER_PATTERN = re.compile(r"\[(?:TODO|TBD|PLACEHOLDER)\]|<TODO>", re.I)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
JOINED_WORD_PATTERN = re.compile(r"\b(?:such asweb|isalso|isJapan)\b", re.I)


@dataclass(frozen=True)
class ValidationFinding:
    example_id: str
    code: str
    message: str
    severity: str = "error"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_record_hash(record: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(record).encode("utf-8"))


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalized_components(record: dict[str, Any]) -> dict[str, str]:
    instruction = normalize_for_comparison(str(record.get("instruction", "")))
    input_text = normalize_for_comparison(str(record.get("input", "")))
    response = normalize_for_comparison(str(record.get("response", "")))
    return {
        "instruction": instruction,
        "input": input_text,
        "response": response,
        "instruction_input": f"{instruction}\n{input_text}",
        "instruction_input_response": f"{instruction}\n{input_text}\n{response}",
    }


def _sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w'-]+", text, flags=re.UNICODE))


def _list_counts(text: str) -> tuple[int, int]:
    bullets = len(re.findall(r"(?m)^\s*[-*+]\s+", text))
    ordered = len(re.findall(r"(?m)^\s*\d+[.)]\s+", text))
    return bullets, ordered


def _valid_http_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_format_constraint(record: dict[str, Any]) -> list[ValidationFinding]:
    example_id = str(record.get("example_id", "<missing>"))
    response = str(record.get("response", ""))
    constraint = record.get("format_constraints")
    if not isinstance(constraint, dict):
        return [ValidationFinding(example_id, "invalid_format_constraint", "format_constraints must be an object")]
    constraint_type = constraint.get("type")
    if constraint_type not in FORMAT_TYPES:
        return [ValidationFinding(example_id, "unsupported_format_constraint", f"unsupported format type: {constraint_type!r}")]
    findings: list[ValidationFinding] = []
    bullets, ordered = _list_counts(response)
    if constraint_type == "exact_bullets" and bullets != constraint.get("count"):
        findings.append(ValidationFinding(example_id, "wrong_bullet_count", f"expected {constraint.get('count')} bullets; found {bullets}"))
    elif constraint_type == "exact_sentences" and _sentence_count(response) != constraint.get("count"):
        findings.append(ValidationFinding(example_id, "wrong_sentence_count", f"expected {constraint.get('count')} sentences; found {_sentence_count(response)}"))
    elif constraint_type == "maximum_words" and _word_count(response) > int(constraint.get("count", -1)):
        findings.append(ValidationFinding(example_id, "maximum_words_exceeded", "response exceeds configured maximum word count"))
    elif constraint_type == "one_word" and _word_count(response) != 1:
        findings.append(ValidationFinding(example_id, "one_word_violation", f"expected one word; found {_word_count(response)}"))
    elif constraint_type == "exact_words" and _word_count(response) != int(constraint.get("count", -1)):
        findings.append(ValidationFinding(example_id, "wrong_word_count", f"expected {constraint.get('count')} words; found {_word_count(response)}"))
    elif constraint_type == "exact_numbered_items" and ordered != int(constraint.get("count", -1)):
        findings.append(ValidationFinding(example_id, "wrong_numbered_item_count", f"expected {constraint.get('count')} numbered items; found {ordered}"))
    elif constraint_type == "exact_labeled_fields":
        labels = constraint.get("labels", [])
        present = [label for label in labels if re.search(rf"(?m)^\s*{re.escape(str(label))}\s*:\s*\S+", response)]
        if len(present) != len(labels):
            findings.append(ValidationFinding(example_id, "missing_labeled_fields", "response is missing one or more required labelled fields"))
    elif constraint_type in {"json_object", "required_fields"}:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            findings.append(ValidationFinding(example_id, "invalid_json", "response must be a parseable JSON object"))
        else:
            required = set(constraint.get("required_keys", []))
            missing = sorted(required - set(parsed))
            if missing:
                findings.append(ValidationFinding(example_id, "missing_json_keys", f"missing required keys: {missing}"))
            if constraint.get("strict", False) and set(parsed) != required:
                findings.append(ValidationFinding(example_id, "extra_json_keys", "strict JSON response contains unexpected keys"))
            types = constraint.get("value_types", {})
            type_map = {"string": str, "number": (int, float), "boolean": bool, "array": list, "object": dict, "null": type(None)}
            for key, expected in types.items():
                if key in parsed and expected in type_map and not isinstance(parsed[key], type_map[expected]):
                    findings.append(ValidationFinding(example_id, "wrong_json_value_type", f"key {key!r} must have type {expected}"))
        if "```" in response and not constraint.get("allow_markdown_fence", False):
            findings.append(ValidationFinding(example_id, "json_markdown_fence", "JSON response must not use a markdown fence"))
    elif constraint_type == "ordered_list" and ordered == 0:
        findings.append(ValidationFinding(example_id, "ordered_list_required", "response has no ordered-list items"))
    elif constraint_type == "unordered_list" and bullets == 0:
        findings.append(ValidationFinding(example_id, "unordered_list_required", "response has no bullet-list items"))
    elif constraint_type == "no_list" and (bullets or ordered):
        findings.append(ValidationFinding(example_id, "list_forbidden", "response contains list structure"))
    elif constraint_type == "required_headings":
        missing = [heading for heading in constraint.get("headings", []) if not re.search(rf"(?m)^\s*{re.escape(heading)}\s*:?\s*$", response)]
        if missing:
            findings.append(ValidationFinding(example_id, "missing_headings", f"missing headings: {missing}"))
    elif constraint_type == "plain_text_only" and re.search(r"(?m)```|^\s*#", response):
        findings.append(ValidationFinding(example_id, "plain_text_violation", "response contains markdown markup"))
    return findings


def validate_record(record: Any) -> list[ValidationFinding]:
    if not isinstance(record, dict):
        return [ValidationFinding("<missing>", "invalid_record", "record must be a JSON object")]
    example_id = str(record.get("example_id", "<missing>"))
    findings: list[ValidationFinding] = []

    def error(code: str, message: str) -> None:
        findings.append(ValidationFinding(example_id, code, message))

    required = {
        "schema_version", "example_id", "capability", "instruction", "input",
        "response", "source_type", "source_reference", "language", "difficulty",
        "answer_style", "format_constraints", "facts", "quality", "metadata",
    }
    missing = sorted(required - set(record))
    if missing:
        error("missing_fields", f"missing required fields: {missing}")
        return findings
    unknown = sorted(set(record) - required)
    if unknown:
        error("unsupported_fields", f"unsupported fields: {unknown}")
    if record["schema_version"] != SCHEMA_VERSION:
        error("unsupported_schema_version", f"expected {SCHEMA_VERSION}")
    if not re.fullmatch(r"viq1_(?:\d{6}|b\d{3}_\d{6})", str(record["example_id"])):
        error("invalid_example_id", "example_id must match viq1_000001 or viq1_b001_000001")
    if record["capability"] not in CAPABILITIES:
        error("unsupported_capability", f"unsupported capability: {record['capability']!r}")
    for field in ("instruction", "response"):
        if not isinstance(record[field], str) or not record[field].strip():
            error(f"empty_{field}", f"{field} must be a non-empty string")
    if not isinstance(record["input"], str):
        error("invalid_input", "input must be a string")
    for field in ("instruction", "input", "response"):
        value = record[field] if isinstance(record[field], str) else ""
        if "\r" in value:
            error("non_normalized_line_endings", f"{field} contains CR line endings")
        if CONTROL_PATTERN.search(value):
            error("control_character", f"{field} contains a hidden control character")
        if PLACEHOLDER_PATTERN.search(value):
            error("placeholder_text", f"{field} contains unresolved placeholder text")
    if any(marker in str(record["response"]) for marker in TEMPLATE_MARKERS):
        error("template_marker", "response contains an instruction template marker")
    if JOINED_WORD_PATTERN.search(str(record["response"])):
        error("joined_word", "response contains a known joined-word defect")
    if str(record["response"]).rstrip().endswith((",", ";", ":", "-")):
        error("possibly_truncated", "response ends with incomplete punctuation")
    words = normalize_for_comparison(str(record["response"])).split()
    if words and 1 - len(set(words)) / len(words) > 0.45:
        error("response_repetition", "response has excessive repeated words")
    if record["source_type"] not in SOURCE_TYPES:
        error("unsupported_source_type", "unsupported source_type")
    if record["language"] != "en":
        error("unsupported_language", "only English is supported in v1")
    if record["difficulty"] not in DIFFICULTIES:
        error("unsupported_difficulty", "unsupported difficulty")
    quality = record["quality"]
    if not isinstance(quality, dict) or quality.get("review_status") not in REVIEW_STATUSES:
        error("invalid_quality", "quality must contain a supported review_status")
    metadata = record["metadata"]
    if not isinstance(metadata, dict) or not all(metadata.get(key) for key in ("created_at", "created_by", "license", "provenance")):
        error("invalid_metadata", "metadata requires created_at, created_by, license, and provenance")
    facts = record["facts"]
    if not isinstance(facts, list):
        error("invalid_facts", "facts must be a list")
    if record["capability"] == "short_factual_qa":
        if not _valid_http_url(record.get("source_reference")):
            error("invalid_source_reference", "factual examples require an HTTP(S) source_reference")
        if not facts:
            error("factual_verification_missing", "factual examples require at least one verified fact")
        for fact in facts if isinstance(facts, list) else []:
            if not isinstance(fact, dict) or fact.get("verification_status") != "verified" or not _valid_http_url(fact.get("verification_source")):
                error("factual_verification_missing", "every factual claim requires verified status and source")
            if fact.get("time_sensitive") and not fact.get("reference_date"):
                error("factual_reference_date_missing", "time-sensitive facts require reference_date")
    findings.extend(validate_format_constraint(record))
    return findings


def validate_records(records: list[dict[str, Any]]) -> list[ValidationFinding]:
    findings = [finding for record in records for finding in validate_record(record)]
    ids = [str(record.get("example_id", "<missing>")) for record in records]
    for example_id, count in Counter(ids).items():
        if count > 1:
            findings.append(ValidationFinding(example_id, "duplicate_example_id", f"example_id appears {count} times"))
    return findings


def exact_duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for field in normalized_components({}).keys():
        values: dict[str, list[str]] = defaultdict(list)
        for record in records:
            value = normalized_components(record)[field]
            if value:
                values[value].append(record["example_id"])
        for value, ids in sorted(values.items()):
            if len(ids) > 1:
                groups.append({"field": field, "normalized_sha256": sha256_bytes(value.encode("utf-8")), "example_ids": sorted(ids)})
    return groups


def _token_set(record: dict[str, Any]) -> set[str]:
    combined = normalized_components(record)["instruction_input_response"]
    words = combined.split()
    return {" ".join(words[index:index + 3]) for index in range(max(0, len(words) - 2))} or set(words)


def near_duplicate_candidates(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1]")
    token_sets = {record["example_id"]: _token_set(record) for record in records}
    candidates = []
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            left_set = token_sets[left["example_id"]]
            right_set = token_sets[right["example_id"]]
            union = left_set | right_set
            similarity = len(left_set & right_set) / len(union) if union else 1.0
            if similarity >= threshold:
                candidates.append({"left": left["example_id"], "right": right["example_id"], "similarity": round(similarity, 6)})
    return sorted(candidates, key=lambda row: (-row["similarity"], row["left"], row["right"]))


def quality_score(record: dict[str, Any], findings: list[ValidationFinding], duplicate_risk: bool) -> dict[str, Any]:
    score = 100
    reasons = []
    for finding in findings:
        deduction = 20 if finding.severity == "error" else 5
        score -= deduction
        reasons.append({"code": finding.code, "points": -deduction})
    if duplicate_risk:
        score -= 15
        reasons.append({"code": "duplicate_risk", "points": -15})
    word_count = _word_count(str(record.get("response", "")))
    if word_count < 2:
        score -= 10
        reasons.append({"code": "very_short_response", "points": -10})
    elif word_count > 100:
        score -= 10
        reasons.append({"code": "very_long_response", "points": -10})
    return {"score": max(0, score), "reasons": reasons, "automatic_approval": False}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at line {line_number}: {error}") from error
            records.append(record)
    return records


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records))


def load_review_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    decisions = load_jsonl(path)
    result = {}
    for decision in decisions:
        example_id = decision.get("example_id")
        if example_id in result:
            raise ValueError(f"duplicate review decision: {example_id}")
        if decision.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported review schema for {example_id}")
        if decision.get("status") not in REVIEW_STATUSES:
            raise ValueError(f"unsupported review status for {example_id}")
        result[example_id] = decision
    return result


def validate_review_decisions(records: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]) -> list[ValidationFinding]:
    by_id = {record["example_id"]: record for record in records}
    findings = []
    for example_id, decision in decisions.items():
        if example_id not in by_id:
            findings.append(ValidationFinding(example_id, "orphan_review", "review decision has no source record"))
            continue
        if decision.get("source_sha256") != source_record_hash(by_id[example_id]):
            findings.append(ValidationFinding(example_id, "stale_review", "review decision source hash is stale"))
        if not decision.get("reviewer") or not decision.get("reviewed_at"):
            findings.append(ValidationFinding(example_id, "incomplete_review", "reviewer and reviewed_at are required"))
    return findings


def transition_allowed(current: str, target: str) -> bool:
    return target in TRANSITIONS.get(current, set())


def make_review_decision(record: dict[str, Any], status: str, reviewer: str, notes: str | None) -> dict[str, Any]:
    if status not in REVIEW_STATUSES or status == "unreviewed":
        raise ValueError("set-status must be a non-unreviewed supported status")
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "example_id": record["example_id"],
        "status": status,
        "reviewer": reviewer,
        "notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_record_hash(record),
    }


def deterministic_split(records: list[dict[str, Any]], validation_ratio: float, seed: int) -> dict[str, str]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    if len(records) < 2:
        raise ValueError("at least two approved records are required")
    by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_capability[record["capability"]].append(record)
    assignments = {}
    for capability in CAPABILITIES:
        rows = sorted(by_capability.get(capability, []), key=lambda record: sha256_bytes(f"{seed}:{record['example_id']}".encode("utf-8")))
        validation_count = max(1, round(len(rows) * validation_ratio)) if len(rows) >= 2 else 0
        for index, record in enumerate(rows):
            assignments[record["example_id"]] = "validation" if index < validation_count else "train"
    if "train" not in assignments.values() or "validation" not in assignments.values():
        raise ValueError("deterministic split produced an empty train or validation split")
    return assignments


def cross_split_leakage(records: list[dict[str, Any]], assignments: dict[str, str], threshold: float) -> list[dict[str, Any]]:
    findings = []
    exact = exact_duplicate_groups(records)
    for group in exact:
        splits = {assignments[example_id] for example_id in group["example_ids"]}
        if len(splits) > 1:
            findings.append({"type": "exact", **group})
    for pair in near_duplicate_candidates(records, threshold):
        if assignments[pair["left"]] != assignments[pair["right"]]:
            findings.append({"type": "near", **pair})
    return findings


def validation_report(records: list[dict[str, Any]], decisions: dict[str, dict[str, Any]], target_distribution: dict[str, float], near_threshold: float) -> dict[str, Any]:
    findings = validate_records(records)
    review_findings = validate_review_decisions(records, decisions)
    exact = exact_duplicate_groups(records)
    near = near_duplicate_candidates(records, near_threshold)
    finding_ids = {finding.example_id for finding in findings}
    duplicate_ids = {example_id for group in exact for example_id in group["example_ids"]}
    category_counts = Counter(record.get("capability") for record in records)
    statuses = Counter(decisions.get(record.get("example_id"), {}).get("status", "unreviewed") for record in records)
    finding_counts = Counter(finding.code for finding in findings)
    scores = {
        record["example_id"]: quality_score(record, [finding for finding in findings if finding.example_id == record["example_id"]], record["example_id"] in duplicate_ids)
        for record in records if "example_id" in record
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "total_records": len(records),
        "valid_records": len(records) - len(finding_ids),
        "invalid_records": len(finding_ids),
        "review_status_counts": dict(statuses),
        "category_counts": dict(category_counts),
        "distribution": {
            capability: {
                "target": target_distribution[capability],
                "actual": category_counts[capability] / len(records) if records else 0.0,
            }
            for capability in CAPABILITIES
        },
        "findings": [finding.__dict__ for finding in findings],
        "finding_counts": dict(sorted(finding_counts.items())),
        "review_findings": [finding.__dict__ for finding in review_findings],
        "exact_duplicate_groups": exact,
        "near_duplicate_candidates": near,
        # Cross-split checks are a release-stage gate because only approved
        # records receive deterministic split assignments.
        "cross_split_duplicate_findings": [],
        "format_constraint_failures": sum(
            count
            for code, count in finding_counts.items()
            if code
            in {
                "wrong_bullet_count",
                "wrong_sentence_count",
                "maximum_words_exceeded",
                "one_word_violation",
                "wrong_word_count",
                "wrong_numbered_item_count",
                "missing_labeled_fields",
                "invalid_json",
                "missing_json_keys",
                "extra_json_keys",
                "wrong_json_value_type",
                "json_markdown_fence",
                "ordered_list_required",
                "unordered_list_required",
                "list_forbidden",
                "missing_headings",
                "plain_text_violation",
            }
        ),
        "factual_verification_gaps": finding_counts["factual_verification_missing"]
        + finding_counts["factual_reference_date_missing"],
        "truncated_responses": finding_counts["possibly_truncated"],
        "template_marker_findings": finding_counts["template_marker"],
        "very_short_answers": sum(
            _word_count(str(record.get("response", ""))) < 2 for record in records
        ),
        "very_long_answers": sum(
            _word_count(str(record.get("response", ""))) > 100 for record in records
        ),
        "response_repetition_findings": finding_counts["response_repetition"],
        "json_parse_failures": finding_counts["invalid_json"],
        "unsupported_fields": finding_counts["unsupported_fields"],
        "quality_scores": scores,
        "automatic_approval": False,
    }


def build_release(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("training_authorized") is not False:
        raise ValueError("training_authorized must remain false")
    source_path = Path(config["source_path"])
    review_path = Path(config["review_path"])
    tokenizer_path = Path(config["tokenizer_path"])
    if sha256_file(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    actual_source_hash = sha256_file(source_path)
    expected_source_hash = config.get("source_sha256")
    if expected_source_hash and actual_source_hash != expected_source_hash:
        raise ValueError("source SHA-256 mismatch")
    records = load_jsonl(source_path)
    decisions = load_review_decisions(review_path)
    findings = validate_records(records) + validate_review_decisions(records, decisions)
    if findings:
        raise ValueError(f"release validation failed: {findings[0].code}: {findings[0].message}")
    approved = [record for record in records if decisions.get(record["example_id"], {}).get("status") == "approved"]
    if not approved:
        raise ValueError("release contains no human-approved records")
    if exact_duplicate_groups(approved):
        raise ValueError("approved release contains exact duplicates")
    if near_duplicate_candidates(
        approved, float(config["near_duplicate_threshold"])
    ):
        raise ValueError("approved release contains high-confidence near duplicates")
    assignments = deterministic_split(approved, float(config["validation_ratio"]), int(config["split_seed"]))
    leakage = cross_split_leakage(approved, assignments, float(config["near_duplicate_threshold"]))
    if leakage:
        raise ValueError("cross-split duplicate leakage detected")
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    pad_id = tokenizer.tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    train_records = sorted(
        [record for record in approved if assignments[record["example_id"]] == "train"],
        key=lambda record: record["example_id"],
    )
    validation_records = sorted(
        [record for record in approved if assignments[record["example_id"]] == "validation"],
        key=lambda record: record["example_id"],
    )

    def pack(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, Any]:
        examples = [
            {
                "instruction": record["instruction"],
                "input": record["input"],
                "output": record["response"],
            }
            for record in rows
        ]
        return build_packed_records(
            examples,
            tokenizer,
            int(config["sequence_length"]),
            pad_id,
            eos_id,
        )

    train_tokens, train_mask, train_stats = pack(train_records)
    validation_tokens, validation_mask, validation_stats = pack(
        validation_records
    )
    if train_stats.truncated_examples or validation_stats.truncated_examples:
        raise ValueError("approved release contains truncated responses")
    tokens = np.concatenate((train_tokens, validation_tokens), axis=0)
    mask = np.concatenate((train_mask, validation_mask), axis=0)
    ordered = [*train_records, *validation_records]
    token_path = Path(config["token_path"])
    mask_path = Path(config["mask_path"])
    release_path = Path(config["release_path"])
    manifest_path = Path(config["release_manifest_path"])
    write_jsonl(release_path, ordered)
    for path, array in ((token_path, tokens.reshape(-1)), (mask_path, mask.reshape(-1))):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        array.tofile(temporary)
        os.replace(temporary, path)
    manifest = {
        "release_version": RELEASE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "training_authorized": False,
        "source_path": str(source_path),
        "source_sha256": actual_source_hash,
        "review_path": str(review_path),
        "review_sha256": sha256_file(review_path),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "approved_example_ids": [record["example_id"] for record in ordered],
        "split_assignments": assignments,
        "split_counts": dict(Counter(assignments.values())),
        "train_packed_records": int(train_tokens.shape[0]),
        "validation_packed_records": int(validation_tokens.shape[0]),
        "capability_counts": dict(Counter(record["capability"] for record in ordered)),
        "token_path": str(token_path),
        "token_sha256": sha256_file(token_path),
        "mask_path": str(mask_path),
        "mask_sha256": sha256_file(mask_path),
        "records": int(tokens.shape[0]),
        "total_tokens": int(tokens.size),
        "assistant_loss_tokens": int(mask.sum()),
        "prompt_tokens": train_stats.prompt_tokens + validation_stats.prompt_tokens,
        "padding_tokens": int((tokens == pad_id).sum()),
        "eos_tokens": train_stats.eos_tokens + validation_stats.eos_tokens,
        "truncated_examples": 0,
        "near_duplicate_threshold": float(config["near_duplicate_threshold"]),
        "cross_split_leakage": [],
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
