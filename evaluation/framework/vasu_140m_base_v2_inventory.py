"""Versioned inventory contracts for VASU-140M base evaluation v2."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from evaluation.framework.vasu_140m_base_v2 import (
    DIMENSIONS,
    INTERFACES,
    RESULT_MODES,
    TOKENIZER_SHA256,
    canonical_json,
    sha256_file,
)
from evaluation.framework.vasu_140m_base_v2_tasks import (
    task_identity,
    validate_task,
)


INVENTORY_SCHEMA_ID = "vasu_140m_base_evaluation_inventory_v2"
PAYLOAD_RECORD_SCHEMA_ID = "vasu_140m_base_evaluation_payload_record_v2"
PROVENANCE_RECORD_SCHEMA_ID = "vasu_140m_base_evaluation_provenance_record_v2"
CONTAMINATION_RECORD_SCHEMA_ID = (
    "vasu_140m_base_evaluation_contamination_record_v2"
)
SPLITS = frozenset({"development", "held_out"})
SHA256_HEX_LENGTH = 64
CONTENT_FIELDS = {
    "likelihood": {
        "context",
        "target",
        "text_sha256",
        "target_sha256",
    },
    "factuality": {"prompt", "prompt_sha256", "choices"},
    "arithmetic": {"prompt", "prompt_sha256"},
    "repetition": {"prompt", "prompt_sha256"},
    "robustness": {
        "baseline_prompt",
        "baseline_prompt_sha256",
        "variant_prompt",
        "variant_prompt_sha256",
    },
    "manual_review": {"prompt", "prompt_sha256"},
}


def inventory_identity(manifest: Mapping[str, object]) -> str:
    """Return an inventory identity excluding its self-hash."""

    body = dict(manifest)
    body.pop("inventory_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def normalized_text_sha256(value: str) -> str:
    """Hash NFC-normalized, whitespace-collapsed contamination text."""

    normalized = unicodedata.normalize(
        "NFC", " ".join(value.strip().split())
    ).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalized_word_count(value: str) -> int:
    """Return the whitespace-token span used by exact contamination scans."""

    return len(unicodedata.normalize("NFC", value).strip().split())


def content_text_sha256(value: str) -> str:
    """Hash exact NFC prompt content while preserving meaningful whitespace."""

    canonical = unicodedata.normalize("NFC", value.replace("\r\n", "\n"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
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


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return text


def _url(value: object, label: str) -> str:
    text = _string(value, label)
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
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


def _bound_file(value: object, label: str) -> tuple[str, str, int, int]:
    binding = _mapping(value, label)
    _exact_keys(binding, {"path", "sha256", "record_count", "byte_count"}, label)
    return (
        _safe_path(binding["path"], f"{label}.path"),
        _sha256(binding["sha256"], f"{label}.sha256"),
        _positive_int(binding["record_count"], f"{label}.record_count"),
        _positive_int(binding["byte_count"], f"{label}.byte_count"),
    )


def _validate_commitment(value: object, index: int) -> str:
    label = f"item commitment {index}"
    commitment = _mapping(value, label)
    _exact_keys(
        commitment,
        {"item_id", "task_sha256", "provenance_sha256"},
        label,
    )
    item_id = _string(commitment["item_id"], f"{label}.item_id")
    _sha256(commitment["task_sha256"], f"{label}.task_sha256")
    _sha256(commitment["provenance_sha256"], f"{label}.provenance_sha256")
    return item_id


def validate_inventory_manifest(manifest: Mapping[str, object]) -> None:
    """Validate one exact development or sealed held-out inventory."""

    _exact_keys(
        manifest,
        {
            "schema_id",
            "inventory_id",
            "suite_id",
            "repository_commit",
            "dimension",
            "split",
            "interface",
            "tokenizer_sha256",
            "payload",
            "provenance_index",
            "contamination_index",
            "scorer",
            "generation_profile_ids",
            "item_commitments",
            "access",
            "fixture_only",
            "production_suite_frozen",
            "evaluation_run_authorized",
            "training_authorized",
            "inventory_sha256",
        },
        "inventory manifest",
    )
    if manifest["schema_id"] != INVENTORY_SCHEMA_ID:
        raise ValueError("inventory schema identity mismatch")
    _string(manifest["inventory_id"], "inventory_id")
    _string(manifest["suite_id"], "suite_id")
    commit = _string(manifest["repository_commit"], "repository_commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("repository_commit must be a lowercase Git commit")
    dimension = _string(manifest["dimension"], "dimension")
    split = _string(manifest["split"], "split")
    if dimension not in DIMENSIONS or split not in SPLITS:
        raise ValueError("inventory dimension or split is unsupported")
    if manifest["interface"] != INTERFACES[dimension]:
        raise ValueError("inventory interface does not match its dimension")
    if _sha256(manifest["tokenizer_sha256"], "tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("inventory tokenizer identity mismatch")

    payload_path, _, payload_count, _ = _bound_file(manifest["payload"], "payload")
    provenance_path, _, provenance_count, _ = _bound_file(
        manifest["provenance_index"], "provenance_index"
    )
    contamination_path, _, contamination_count, _ = _bound_file(
        manifest["contamination_index"], "contamination_index"
    )
    scorer = _mapping(manifest["scorer"], "scorer")
    _exact_keys(scorer, {"path", "sha256"}, "scorer")
    scorer_path = _safe_path(scorer["path"], "scorer.path")
    _sha256(scorer["sha256"], "scorer.sha256")
    if len({payload_path, provenance_path, contamination_path, scorer_path}) != 4:
        raise ValueError("inventory artifact paths must be distinct")

    commitments = manifest["item_commitments"]
    if not isinstance(commitments, list) or not commitments:
        raise ValueError("item_commitments must be a non-empty list")
    item_ids = [_validate_commitment(item, index) for index, item in enumerate(commitments)]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item commitment IDs must be unique")
    if not (
        payload_count == provenance_count == contamination_count == len(item_ids)
    ):
        raise ValueError("inventory artifact record counts must match commitments")

    profiles = manifest["generation_profile_ids"]
    if not isinstance(profiles, list) or any(not isinstance(item, str) for item in profiles):
        raise ValueError("generation_profile_ids must be a string list")
    if len(profiles) != len(set(profiles)):
        raise ValueError("generation_profile_ids must be unique")
    expected_modes = RESULT_MODES[dimension]
    if expected_modes == {"direct_likelihood"} or expected_modes == {"manual"}:
        if profiles:
            raise ValueError("non-generation inventory cannot bind profiles")
    elif set(profiles) != {"greedy-v1", "sampled-v1"}:
        raise ValueError("generation inventory must bind greedy-v1 and sampled-v1")

    access = _mapping(manifest["access"], "access")
    _exact_keys(
        access,
        {
            "state",
            "payload_format",
            "encryption_algorithm",
            "recipient_fingerprint",
            "opening_authorized",
        },
        "access",
    )
    if split == "development":
        if access != {
            "state": "available",
            "payload_format": "jsonl",
            "encryption_algorithm": "none",
            "recipient_fingerprint": None,
            "opening_authorized": False,
        }:
            raise ValueError("development access contract is invalid")
    else:
        if (
            access["state"] != "sealed"
            or access["payload_format"] != "encrypted_jsonl"
            or access["encryption_algorithm"] != "age-x25519"
            or not isinstance(access["recipient_fingerprint"], str)
            or not access["recipient_fingerprint"].strip()
            or access["opening_authorized"] is not False
        ):
            raise ValueError("held-out access contract is invalid")
    if not isinstance(manifest["fixture_only"], bool):
        raise ValueError("fixture_only must be boolean")
    for flag, expected in {
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }.items():
        if manifest[flag] is not expected:
            raise ValueError(f"{flag} must be {str(expected).lower()}")
    if _sha256(manifest["inventory_sha256"], "inventory_sha256") != inventory_identity(manifest):
        raise ValueError("inventory manifest identity mismatch")


def validate_payload_record(record: Mapping[str, object], dimension: str) -> None:
    """Validate one plaintext development payload record."""

    _exact_keys(record, {"schema_id", "task", "content"}, "payload record")
    if record["schema_id"] != PAYLOAD_RECORD_SCHEMA_ID:
        raise ValueError("payload record schema identity mismatch")
    task = _mapping(record["task"], "payload task")
    validate_task(task)
    if task["dimension"] != dimension or task["split"] != "development":
        raise ValueError("payload task dimension/split mismatch")
    content = _mapping(record["content"], "payload content")
    _exact_keys(content, CONTENT_FIELDS[dimension], "payload content")
    inputs = _mapping(task["input"], "task.input")
    if dimension == "likelihood":
        context = content["context"]
        if not isinstance(context, str):
            raise ValueError("payload content.context must be a string")
        target = _string(content["target"], "payload content.target")
        observed = content_text_sha256(context + target)
        target_sha = content_text_sha256(target)
        if content["text_sha256"] != observed or inputs["text_sha256"] != observed:
            raise ValueError("likelihood text identity mismatch")
        if content["target_sha256"] != target_sha:
            raise ValueError("likelihood target identity mismatch")
    elif dimension == "factuality":
        prompt = _string(content["prompt"], "payload content.prompt")
        observed = content_text_sha256(prompt)
        if content["prompt_sha256"] != observed or inputs["prompt_sha256"] != observed:
            raise ValueError("factuality prompt identity mismatch")
        choices = content["choices"]
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError("factuality choices must be a non-empty list")
        choice_ids: list[str] = []
        for index, raw in enumerate(choices):
            choice = _mapping(raw, f"choice {index}")
            _exact_keys(choice, {"choice_id", "text"}, f"choice {index}")
            choice_ids.append(_string(choice["choice_id"], "choice_id"))
            _string(choice["text"], "choice.text")
        if choice_ids != inputs["choice_ids"]:
            raise ValueError("factuality choice identity/order mismatch")
    elif dimension == "robustness":
        for prefix in ("baseline", "variant"):
            text = _string(content[f"{prefix}_prompt"], f"{prefix}_prompt")
            observed = content_text_sha256(text)
            if (
                content[f"{prefix}_prompt_sha256"] != observed
                or inputs[f"{prefix}_prompt_sha256"] != observed
            ):
                raise ValueError(f"robustness {prefix} prompt identity mismatch")
    else:
        prompt = _string(content["prompt"], "payload content.prompt")
        observed = content_text_sha256(prompt)
        if content["prompt_sha256"] != observed or inputs["prompt_sha256"] != observed:
            raise ValueError("payload prompt identity mismatch")


def provenance_identity(record: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(record)).hexdigest()


def validate_provenance_record(record: Mapping[str, object]) -> None:
    """Validate public provenance without requiring prompt disclosure."""

    _exact_keys(
        record,
        {
            "schema_id",
            "item_id",
            "source_name",
            "source_url",
            "license_name",
            "license_url",
            "source_revision",
            "citation",
            "parent_document_id",
            "retrieved_at",
            "human_authored",
        },
        "provenance record",
    )
    if record["schema_id"] != PROVENANCE_RECORD_SCHEMA_ID:
        raise ValueError("provenance schema identity mismatch")
    for field in (
        "item_id",
        "source_name",
        "license_name",
        "source_revision",
        "citation",
        "parent_document_id",
    ):
        _string(record[field], f"provenance.{field}")
    _url(record["source_url"], "provenance.source_url")
    _url(record["license_url"], "provenance.license_url")
    _timestamp(record["retrieved_at"], "provenance.retrieved_at")
    if not isinstance(record["human_authored"], bool):
        raise ValueError("provenance.human_authored must be boolean")


def validate_contamination_record(record: Mapping[str, object]) -> None:
    """Validate a public hash-only contamination commitment."""

    _exact_keys(
        record,
        {
            "schema_id",
            "item_id",
            "parent_document_id",
            "prompt_exact_commitments",
            "answer_exact_commitments",
            "ngram_words",
            "ngram_sha256s",
            "semantic_fingerprint",
        },
        "contamination record",
    )
    if record["schema_id"] != CONTAMINATION_RECORD_SCHEMA_ID:
        raise ValueError("contamination schema identity mismatch")
    _string(record["item_id"], "contamination.item_id")
    _string(record["parent_document_id"], "contamination.parent_document_id")
    for field in ("prompt_exact_commitments", "answer_exact_commitments"):
        values = record[field]
        if (
            not isinstance(values, list)
            or (field == "prompt_exact_commitments" and not values)
        ):
            raise ValueError(f"contamination.{field} must be a commitment list")
        identities: list[str] = []
        for index, raw in enumerate(values):
            commitment = _mapping(raw, f"contamination.{field}[{index}]")
            _exact_keys(
                commitment,
                {"sha256", "word_count"},
                f"contamination.{field}[{index}]",
            )
            identities.append(
                _sha256(
                    commitment["sha256"],
                    f"contamination.{field}[{index}].sha256",
                )
            )
            _positive_int(
                commitment["word_count"],
                f"contamination.{field}[{index}].word_count",
            )
        if len(identities) != len(set(identities)):
            raise ValueError(f"contamination.{field} contains duplicate hashes")
    values = record["ngram_sha256s"]
    if not isinstance(values, list) or len(values) != len(set(values)):
        raise ValueError("contamination.ngram_sha256s must be a unique hash list")
    for value in values:
        _sha256(value, "contamination.ngram_sha256s")
    if (
        isinstance(record["ngram_words"], bool)
        or not isinstance(record["ngram_words"], int)
        or record["ngram_words"] < 8
    ):
        raise ValueError("contamination.ngram_words must be at least 8")
    fingerprint = _mapping(record["semantic_fingerprint"], "semantic_fingerprint")
    _exact_keys(fingerprint, {"method", "value"}, "semantic_fingerprint")
    _string(fingerprint["method"], "semantic_fingerprint.method")
    _sha256(fingerprint["value"], "semantic_fingerprint.value")


def _contamination_values(
    payload: Mapping[str, object], dimension: str
) -> tuple[list[str], list[str]]:
    content = _mapping(payload["content"], "payload content")
    task = _mapping(payload["task"], "payload task")
    scoring = _mapping(task["scoring"], "task.scoring")
    if dimension == "likelihood":
        return [str(content["context"])], [str(content["target"])]
    if dimension == "factuality":
        correct_id = scoring["correct_choice_id"]
        choices = content["choices"]
        assert isinstance(choices, list)
        answers = [
            str(choice["text"])
            for choice in choices
            if isinstance(choice, Mapping) and choice["choice_id"] == correct_id
        ]
        return [str(content["prompt"])], answers
    if dimension == "arithmetic":
        return [str(content["prompt"])], [str(scoring["expected_answer"])]
    if dimension == "robustness":
        answers = scoring["accepted_answers"]
        assert isinstance(answers, list)
        return [
            str(content["baseline_prompt"]),
            str(content["variant_prompt"]),
        ], [str(value) for value in answers]
    return [str(content["prompt"])], []


def _expected_exact_commitments(values: list[str]) -> list[dict[str, object]]:
    unique = {
        normalized_text_sha256(value): normalized_word_count(value) for value in values
    }
    return [
        {"sha256": digest, "word_count": unique[digest]}
        for digest in sorted(unique)
    ]


def _expected_ngram_hashes(values: list[str], width: int) -> list[str]:
    hashes: set[str] = set()
    for value in values:
        words = unicodedata.normalize("NFC", value).strip().split()
        for index in range(max(0, len(words) - width + 1)):
            hashes.add(
                normalized_text_sha256(" ".join(words[index : index + width]))
            )
    return sorted(hashes)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object")
        records.append(value)
    return records


def validate_inventory_manifest_files(
    manifest: Mapping[str, object], repository_root: Path, *, open_held_out: bool = False
) -> None:
    """Validate bound bytes while refusing to decrypt held-out payloads."""

    validate_inventory_manifest(manifest)
    if open_held_out:
        raise PermissionError("held-out opening requires a future accepted decision")
    root = repository_root.resolve()
    bindings = {
        name: _bound_file(manifest[name], name)
        for name in ("payload", "provenance_index", "contamination_index")
    }
    scorer = _mapping(manifest["scorer"], "scorer")
    scorer_binding = (
        _safe_path(scorer["path"], "scorer.path"),
        _sha256(scorer["sha256"], "scorer.sha256"),
    )
    for label, (relative, expected, *_) in {
        **bindings,
        "scorer": scorer_binding,
    }.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{label} path is missing or unsafe")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} file identity mismatch")
        binding = bindings.get(label)
        if binding is not None and path.stat().st_size != binding[3]:
            raise ValueError(f"{label} file byte count mismatch")

    if manifest["split"] == "held_out":
        held_out_payload = root / bindings["payload"][0]
        with held_out_payload.open("rb") as handle:
            if handle.readline() != b"age-encryption.org/v1\n":
                raise ValueError("held-out payload does not have an Age v1 header")

    provenance = _load_jsonl(root / bindings["provenance_index"][0])
    contamination = _load_jsonl(root / bindings["contamination_index"][0])
    if len(provenance) != bindings["provenance_index"][2]:
        raise ValueError("provenance record count mismatch")
    if len(contamination) != bindings["contamination_index"][2]:
        raise ValueError("contamination record count mismatch")
    for record in provenance:
        validate_provenance_record(record)
    for record in contamination:
        validate_contamination_record(record)
    commitments = {
        str(item["item_id"]): item for item in manifest["item_commitments"]
    }
    provenance_ids = [str(item["item_id"]) for item in provenance]
    contamination_ids = [str(item["item_id"]) for item in contamination]
    if len(provenance_ids) != len(set(provenance_ids)):
        raise ValueError("provenance IDs must be unique")
    if len(contamination_ids) != len(set(contamination_ids)):
        raise ValueError("contamination IDs must be unique")
    if set(provenance_ids) != set(commitments):
        raise ValueError("provenance IDs do not match commitments")
    if set(contamination_ids) != set(commitments):
        raise ValueError("contamination IDs do not match commitments")
    for record in provenance:
        expected = commitments[str(record["item_id"])]["provenance_sha256"]
        if provenance_identity(record) != expected:
            raise ValueError("provenance record identity mismatch")

    if manifest["split"] == "development":
        payload = _load_jsonl(root / bindings["payload"][0])
        if len(payload) != bindings["payload"][2]:
            raise ValueError("payload record count mismatch")
        for record in payload:
            validate_payload_record(record, str(manifest["dimension"]))
        payload_ids = [str(item["task"]["item_id"]) for item in payload]
        if len(payload_ids) != len(set(payload_ids)):
            raise ValueError("payload IDs must be unique")
        if set(payload_ids) != set(commitments):
            raise ValueError("payload IDs do not match commitments")
        for record in payload:
            expected = commitments[str(record["task"]["item_id"])]["task_sha256"]
            if task_identity(record["task"]) != expected:
                raise ValueError("payload task identity mismatch")
        contamination_by_id = {
            str(record["item_id"]): record for record in contamination
        }
        dimension = str(manifest["dimension"])
        for record in payload:
            item_id = str(record["task"]["item_id"])
            contamination_record = contamination_by_id[item_id]
            prompts, answers = _contamination_values(record, dimension)
            if contamination_record["prompt_exact_commitments"] != (
                _expected_exact_commitments(prompts)
            ):
                raise ValueError("prompt exact commitments do not match payload")
            if contamination_record["answer_exact_commitments"] != (
                _expected_exact_commitments(answers)
            ):
                raise ValueError("answer exact commitments do not match payload")
            width = int(contamination_record["ngram_words"])
            if contamination_record["ngram_sha256s"] != _expected_ngram_hashes(
                [*prompts, *answers], width
            ):
                raise ValueError("n-gram commitments do not match payload")


def validate_numeric_summary(value: object) -> float:
    """Small helper used by inventory qualifications to reject NaN metadata."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("summary value must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("summary value must be finite")
    return number
