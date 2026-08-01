from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_base_v2 import TOKENIZER_SHA256, sha256_file
from evaluation.framework.vasu_140m_base_v2_inventory import (
    CONTAMINATION_RECORD_SCHEMA_ID,
    INVENTORY_SCHEMA_ID,
    PAYLOAD_RECORD_SCHEMA_ID,
    PROVENANCE_RECORD_SCHEMA_ID,
    content_text_sha256,
    inventory_identity,
    normalized_text_sha256,
    provenance_identity,
    validate_contamination_record,
    validate_inventory_manifest,
    validate_inventory_manifest_files,
    validate_numeric_summary,
    validate_payload_record,
    validate_provenance_record,
)
from evaluation.framework.vasu_140m_base_v2_tasks import TASK_SCHEMA_ID, task_identity


def canonical_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def write_file(root: Path, relative: str, payload: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def arithmetic_task(split: str = "development") -> dict[str, object]:
    prompt = "What is 20 plus 22?"
    return {
        "schema_id": TASK_SCHEMA_ID,
        "item_id": "arithmetic-001",
        "dimension": "arithmetic",
        "split": split,
        "strata": {
            "operation": "addition",
            "difficulty": "easy",
            "template_family": "direct",
        },
        "input": {"prompt_sha256": content_text_sha256(prompt)},
        "scoring": {"answer_type": "integer", "expected_answer": "42"},
    }


def payload_record() -> dict[str, object]:
    prompt = "What is 20 plus 22?"
    return {
        "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
        "task": arithmetic_task(),
        "content": {
            "prompt": prompt,
            "prompt_sha256": content_text_sha256(prompt),
        },
    }


def provenance_record() -> dict[str, object]:
    return {
        "schema_id": PROVENANCE_RECORD_SCHEMA_ID,
        "item_id": "arithmetic-001",
        "source_name": "Synthetic qualification fixture",
        "source_url": "https://example.test/source",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "fixture-v1",
        "citation": "Synthetic fixture; no protected benchmark content.",
        "parent_document_id": "fixture-parent-001",
        "retrieved_at": "2026-08-01T12:00:00+05:30",
        "human_authored": True,
    }


def contamination_record() -> dict[str, object]:
    return {
        "schema_id": CONTAMINATION_RECORD_SCHEMA_ID,
        "item_id": "arithmetic-001",
        "parent_document_id": "fixture-parent-001",
        "prompt_exact_commitments": [
            {
                "sha256": normalized_text_sha256("What is 20 plus 22?"),
                "word_count": 5,
            }
        ],
        "answer_exact_commitments": [
            {"sha256": normalized_text_sha256("42"), "word_count": 1}
        ],
        "ngram_words": 8,
        "ngram_sha256s": [],
        "semantic_fingerprint": {
            "method": "fixture-minhash-v1",
            "value": hashlib.sha256(b"fixture-semantic").hexdigest(),
        },
    }


def materialize_inventory(
    root: Path, *, split: str = "development"
) -> dict[str, object]:
    base = f"evaluation/inventories/fixture/{split}/arithmetic"
    payload_path = f"{base}/payload.{'jsonl.age' if split == 'held_out' else 'jsonl'}"
    provenance_path = f"{base}/provenance.jsonl"
    contamination_path = f"{base}/contamination.jsonl"
    scorer_path = "evaluation/framework/vasu_140m_base_v2_tasks.py"
    if split == "development":
        payload = canonical_line(payload_record())
        task_sha = task_identity(arithmetic_task())
    else:
        payload = b"age-encryption.org/v1\nsynthetic opaque payload"
        task_sha = "a" * 64
    payload_file = write_file(root, payload_path, payload)
    provenance = provenance_record()
    provenance_file = write_file(
        root, provenance_path, canonical_line(provenance)
    )
    contamination_file = write_file(
        root, contamination_path, canonical_line(contamination_record())
    )
    scorer_file = write_file(root, scorer_path, "# synthetic scorer binding\n")
    manifest: dict[str, object] = {
        "schema_id": INVENTORY_SCHEMA_ID,
        "inventory_id": f"arithmetic-{split}-fixture-v1",
        "suite_id": "vasu-140m-base-eval-v2-fixture",
        "repository_commit": "a" * 40,
        "dimension": "arithmetic",
        "split": split,
        "interface": "raw_continuation",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "payload": {
            "path": payload_path,
            "sha256": sha256_file(payload_file),
            "record_count": 1,
            "byte_count": payload_file.stat().st_size,
        },
        "provenance_index": {
            "path": provenance_path,
            "sha256": sha256_file(provenance_file),
            "record_count": 1,
            "byte_count": provenance_file.stat().st_size,
        },
        "contamination_index": {
            "path": contamination_path,
            "sha256": sha256_file(contamination_file),
            "record_count": 1,
            "byte_count": contamination_file.stat().st_size,
        },
        "scorer": {"path": scorer_path, "sha256": sha256_file(scorer_file)},
        "generation_profile_ids": ["greedy-v1", "sampled-v1"],
        "item_commitments": [
            {
                "item_id": "arithmetic-001",
                "task_sha256": task_sha,
                "provenance_sha256": provenance_identity(provenance),
            }
        ],
        "access": (
            {
                "state": "available",
                "payload_format": "jsonl",
                "encryption_algorithm": "none",
                "recipient_fingerprint": None,
                "opening_authorized": False,
            }
            if split == "development"
            else {
                "state": "sealed",
                "payload_format": "encrypted_jsonl",
                "encryption_algorithm": "age-x25519",
                "recipient_fingerprint": "AGE-RECIPIENT-FIXTURE-001",
                "opening_authorized": False,
            }
        ),
        "fixture_only": True,
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
        "inventory_sha256": "0" * 64,
    }
    manifest["inventory_sha256"] = inventory_identity(manifest)
    return manifest


def rehash(value: dict[str, object]) -> None:
    value["inventory_sha256"] = inventory_identity(value)


def test_development_inventory_and_files_validate(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    validate_inventory_manifest(value)
    validate_inventory_manifest_files(value, tmp_path)
    assert value["fixture_only"] is True
    assert value["training_authorized"] is False


def test_held_out_payload_remains_opaque_and_cannot_be_opened(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path, split="held_out")
    validate_inventory_manifest(value)
    validate_inventory_manifest_files(value, tmp_path)
    with pytest.raises(PermissionError, match="future accepted decision"):
        validate_inventory_manifest_files(value, tmp_path, open_held_out=True)


@pytest.mark.parametrize("field", ["schema_id", "dimension", "interface"])
def test_manifest_identity_fields_fail_closed(tmp_path: Path, field: str) -> None:
    value = materialize_inventory(tmp_path)
    value[field] = "invalid"
    rehash(value)
    with pytest.raises(ValueError):
        validate_inventory_manifest(value)


def test_manifest_rejects_unknown_and_unhashed_mutation(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_inventory_manifest(value)
    value = materialize_inventory(tmp_path)
    value["suite_id"] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_inventory_manifest(value)


def test_split_access_contracts_cannot_be_substituted(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["access"]["state"] = "sealed"
    rehash(value)
    with pytest.raises(ValueError, match="development access"):
        validate_inventory_manifest(value)
    value = materialize_inventory(tmp_path, split="held_out")
    value["access"]["opening_authorized"] = True
    rehash(value)
    with pytest.raises(ValueError, match="held-out access"):
        validate_inventory_manifest(value)


def test_artifact_counts_and_paths_must_be_consistent(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["payload"]["record_count"] = 2
    rehash(value)
    with pytest.raises(ValueError, match="record counts"):
        validate_inventory_manifest(value)
    value = materialize_inventory(tmp_path)
    value["payload"]["path"] = value["provenance_index"]["path"]
    rehash(value)
    with pytest.raises(ValueError, match="paths must be distinct"):
        validate_inventory_manifest(value)


def test_generation_profiles_follow_dimension_contract(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["generation_profile_ids"] = ["greedy-v1"]
    rehash(value)
    with pytest.raises(ValueError, match="must bind"):
        validate_inventory_manifest(value)
    value = materialize_inventory(tmp_path)
    value["dimension"] = "likelihood"
    value["interface"] = "direct_likelihood"
    value["generation_profile_ids"] = ["greedy-v1", "sampled-v1"]
    rehash(value)
    with pytest.raises(ValueError, match="cannot bind profiles"):
        validate_inventory_manifest(value)


def test_authority_flags_fail_closed(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["production_suite_frozen"] = True
    rehash(value)
    with pytest.raises(ValueError, match="must be false"):
        validate_inventory_manifest(value)


def test_production_candidate_is_structurally_supported_but_not_frozen(
    tmp_path: Path,
) -> None:
    value = materialize_inventory(tmp_path)
    value["fixture_only"] = False
    rehash(value)
    validate_inventory_manifest(value)
    validate_inventory_manifest_files(value, tmp_path)
    assert value["production_suite_frozen"] is False
    assert value["evaluation_run_authorized"] is False
    assert value["training_authorized"] is False


def test_payload_prompt_and_task_identities_are_bound() -> None:
    value = payload_record()
    validate_payload_record(value, "arithmetic")
    value["content"]["prompt"] = "Changed prompt"
    with pytest.raises(ValueError, match="prompt identity"):
        validate_payload_record(value, "arithmetic")


@pytest.mark.parametrize("dimension", ["repetition", "manual_review"])
def test_plain_prompt_payload_dimensions_validate(dimension: str) -> None:
    prompt = f"Synthetic {dimension} prompt"
    value = {
        "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
        "task": {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": f"{dimension}-001",
            "dimension": dimension,
            "split": "development",
            "strata": (
                {"prompt_family": "fixture"}
                if dimension == "repetition"
                else {"category": "fixture"}
            ),
            "input": {"prompt_sha256": content_text_sha256(prompt)},
            "scoring": (
                {"loop_ngram_size": 3}
                if dimension == "repetition"
                else {
                    "rubric_dimensions": [
                        "coherence",
                        "factual_support",
                        "degeneration",
                    ]
                }
            ),
        },
        "content": {
            "prompt": prompt,
            "prompt_sha256": content_text_sha256(prompt),
        },
    }
    validate_payload_record(value, dimension)


def test_likelihood_and_robustness_payloads_validate() -> None:
    text = "Synthetic likelihood target text."
    likelihood = {
        "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
        "task": {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": "likelihood-001",
            "dimension": "likelihood",
            "split": "development",
            "strata": {"source": "fixture"},
            "input": {
                "text_sha256": content_text_sha256(text),
                "target_token_count": 5,
            },
            "scoring": {"kind": "token_log_likelihood"},
        },
        "content": {
            "context": "Synthetic ",
            "target": "likelihood target text.",
            "text_sha256": content_text_sha256(text),
            "target_sha256": content_text_sha256("likelihood target text."),
        },
    }
    validate_payload_record(likelihood, "likelihood")

    baseline = "Synthetic robustness prompt."
    variant = "  Synthetic   robustness prompt.  "
    robustness = {
        "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
        "task": {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": "robustness-001",
            "dimension": "robustness",
            "split": "development",
            "strata": {"variant_kind": "whitespace"},
            "input": {
                "pair_id": "fixture-pair-001",
                "baseline_prompt_sha256": content_text_sha256(baseline),
                "variant_prompt_sha256": content_text_sha256(variant),
                "variant_kind": "whitespace",
            },
            "scoring": {"accepted_answers": ["fixture answer"]},
        },
        "content": {
            "baseline_prompt": baseline,
            "baseline_prompt_sha256": content_text_sha256(baseline),
            "variant_prompt": variant,
            "variant_prompt_sha256": content_text_sha256(variant),
        },
    }
    validate_payload_record(robustness, "robustness")
    assert robustness["task"]["input"]["baseline_prompt_sha256"] != (
        robustness["task"]["input"]["variant_prompt_sha256"]
    )


def test_normalized_text_identity_uses_nfc_and_collapsed_whitespace() -> None:
    assert normalized_text_sha256("café  text") == normalized_text_sha256(
        "cafe\u0301\ntext"
    )
    assert content_text_sha256("café  text") != content_text_sha256(
        "cafe\u0301\ntext"
    )


def test_factual_payload_choice_order_is_bound() -> None:
    prompt = "Fixture factual question?"
    value = {
        "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
        "task": {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": "factuality-001",
            "dimension": "factuality",
            "split": "development",
            "strata": {"task_family": "multiple_choice"},
            "input": {
                "prompt_sha256": content_text_sha256(prompt),
                "choice_ids": ["a", "b"],
            },
            "scoring": {"correct_choice_id": "a"},
        },
        "content": {
            "prompt": prompt,
            "prompt_sha256": content_text_sha256(prompt),
            "choices": [
                {"choice_id": "a", "text": "First"},
                {"choice_id": "b", "text": "Second"},
            ],
        },
    }
    validate_payload_record(value, "factuality")
    value["content"]["choices"].reverse()
    with pytest.raises(ValueError, match="choice identity/order"):
        validate_payload_record(value, "factuality")


def test_provenance_requires_primary_urls_and_timezone() -> None:
    value = provenance_record()
    validate_provenance_record(value)
    value["source_url"] = "relative"
    with pytest.raises(ValueError, match="absolute HTTP"):
        validate_provenance_record(value)
    value = provenance_record()
    value["retrieved_at"] = "2026-08-01"
    with pytest.raises(ValueError, match="timezone"):
        validate_provenance_record(value)


def test_contamination_is_hash_only_and_requires_eight_word_fragments() -> None:
    value = contamination_record()
    validate_contamination_record(value)
    value["ngram_words"] = 7
    with pytest.raises(ValueError, match="at least 8"):
        validate_contamination_record(value)
    value = contamination_record()
    value["answer_exact_commitments"] = [
        {"sha256": "plaintext", "word_count": 1}
    ]
    with pytest.raises(ValueError, match="SHA-256"):
        validate_contamination_record(value)
    value = contamination_record()
    value["prompt_exact_commitments"] *= 2
    with pytest.raises(ValueError, match="duplicate hashes"):
        validate_contamination_record(value)
    value = contamination_record()
    value["answer_exact_commitments"][0]["word_count"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        validate_contamination_record(value)


def test_contamination_identity_collapses_whitespace_and_case() -> None:
    assert normalized_text_sha256("  The   RED Planet ") == normalized_text_sha256(
        "the red planet"
    )


def test_file_mutation_and_commitment_mismatch_fail_closed(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    payload = tmp_path / value["payload"]["path"]
    payload.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="payload file identity mismatch"):
        validate_inventory_manifest_files(value, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("prompt_word_count", "prompt exact commitments"),
        ("answer_hash", "answer exact commitments"),
        ("ngram_extra", "n-gram commitments"),
    ],
)
def test_development_commitments_must_match_plaintext_payload(
    tmp_path: Path, mutation: str, message: str
) -> None:
    value = materialize_inventory(tmp_path)
    contamination_path = tmp_path / value["contamination_index"]["path"]
    record = json.loads(contamination_path.read_text(encoding="utf-8"))
    if mutation == "prompt_word_count":
        record["prompt_exact_commitments"][0]["word_count"] += 1
    elif mutation == "answer_hash":
        record["answer_exact_commitments"][0]["sha256"] = "f" * 64
    else:
        record["ngram_sha256s"] = ["e" * 64]
    contamination_path.write_text(canonical_line(record), encoding="utf-8")
    value["contamination_index"]["sha256"] = sha256_file(contamination_path)
    value["contamination_index"]["byte_count"] = contamination_path.stat().st_size
    rehash(value)
    with pytest.raises(ValueError, match=message):
        validate_inventory_manifest_files(value, tmp_path)


def test_file_byte_count_and_age_header_fail_closed(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["payload"]["byte_count"] += 1
    rehash(value)
    with pytest.raises(ValueError, match="byte count mismatch"):
        validate_inventory_manifest_files(value, tmp_path)
    value = materialize_inventory(tmp_path, split="held_out")
    payload = tmp_path / value["payload"]["path"]
    payload.write_bytes(b"not-age\nopaque")
    value["payload"]["sha256"] = sha256_file(payload)
    value["payload"]["byte_count"] = payload.stat().st_size
    rehash(value)
    with pytest.raises(ValueError, match="Age v1 header"):
        validate_inventory_manifest_files(value, tmp_path)
    value = materialize_inventory(tmp_path)
    value["item_commitments"][0]["provenance_sha256"] = "0" * 64
    rehash(value)
    with pytest.raises(ValueError, match="provenance record identity"):
        validate_inventory_manifest_files(value, tmp_path)


def test_payload_item_set_must_match_commitments(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["item_commitments"][0]["item_id"] = "other"
    rehash(value)
    with pytest.raises(ValueError, match="provenance IDs"):
        validate_inventory_manifest_files(value, tmp_path)


def test_parsed_jsonl_record_count_and_duplicate_ids_fail_closed(
    tmp_path: Path,
) -> None:
    value = materialize_inventory(tmp_path)
    provenance_path = tmp_path / value["provenance_index"]["path"]
    original = provenance_path.read_text(encoding="utf-8")
    provenance_path.write_text(original + original, encoding="utf-8")
    value["provenance_index"]["sha256"] = sha256_file(provenance_path)
    value["provenance_index"]["byte_count"] = provenance_path.stat().st_size
    rehash(value)
    with pytest.raises(ValueError, match="record count mismatch"):
        validate_inventory_manifest_files(value, tmp_path)


def test_unsafe_repository_path_is_rejected(tmp_path: Path) -> None:
    value = materialize_inventory(tmp_path)
    value["payload"]["path"] = "../outside.jsonl"
    rehash(value)
    with pytest.raises(ValueError, match="repository-relative"):
        validate_inventory_manifest(value)


def test_numeric_summary_rejects_boolean_nan_and_infinity() -> None:
    assert validate_numeric_summary(1.5) == 1.5
    for value in (True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            validate_numeric_summary(value)


def test_nested_manifest_mutation_without_rehash_fails(tmp_path: Path) -> None:
    value = deepcopy(materialize_inventory(tmp_path))
    value["access"]["recipient_fingerprint"] = "unexpected"
    with pytest.raises(ValueError):
        validate_inventory_manifest(value)
