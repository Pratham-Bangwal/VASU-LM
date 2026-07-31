from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_base_v2 import (
    DIMENSIONS,
    FAMILY_ID,
    INTERFACES,
    MODEL_CONFIG_SHA256,
    SUITE_SCHEMA_ID,
    TOKENIZER_SHA256,
    build_synthetic_result_manifest,
    inventory_index_identity,
    result_identity,
    resume_identity,
    sha256_file,
    suite_identity,
    validate_result_manifest,
    validate_result_against_suite,
    validate_suite_manifest,
    validate_suite_manifest_files,
)


ROOT = Path(__file__).resolve().parents[1]


def suite_manifest() -> dict[str, object]:
    scorers = [
        {
            "scorer_id": f"{dimension}-scorer-v1",
            "dimension": dimension,
            "metric_kind": "manual" if dimension == "manual_review" else "objective",
            "path": "evaluation/framework/scoring.py",
            "sha256": sha256_file(ROOT / "evaluation/framework/scoring.py"),
        }
        for dimension in sorted(DIMENSIONS)
    ]
    inventories = []
    development_sha = sha256_file(Path(__file__))
    for dimension in sorted(DIMENSIONS):
        for split in ("development", "held_out"):
            inventories.append(
                {
                    "inventory_id": f"{dimension}-{split}-synthetic-v1",
                    "dimension": dimension,
                    "split": split,
                    "path": (
                        "tests/test_vasu_140m_base_evaluation_v2.py"
                        if split == "development"
                        else f"evaluation/held_out/sealed/{dimension}.jsonl"
                    ),
                    "sha256": development_sha if split == "development" else "d" * 64,
                    "provenance_sha256": "e" * 64,
                    "item_count": 2,
                    "interface": INTERFACES[dimension],
                    "scorer_id": f"{dimension}-scorer-v1",
                    "generation_profile_ids": (
                        []
                        if INTERFACES[dimension] == "direct_likelihood"
                        else ["greedy-v1", "sampled-v1"]
                    ),
                    "access_state": "sealed" if split == "held_out" else "available",
                }
            )
    value: dict[str, object] = {
        "schema_id": SUITE_SCHEMA_ID,
        "suite_id": "vasu-140m-base-eval-v2-synthetic",
        "repository_commit": "a" * 40,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer": {
            "path": "assets/tokenizer.json",
            "sha256": TOKENIZER_SHA256,
        },
        "inventories": inventories,
        "scorers": scorers,
        "generation_profiles": [
            {
                "profile_id": "greedy-v1",
                "decoding": "greedy",
                "max_new_tokens": 64,
                "seeds": [],
                "temperature": 0.0,
                "top_k": 0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "eos_token_id": 3,
                "pad_token_id": 0,
                "batch_size": 1,
                "precision": "bf16",
                "kv_cache": True,
            },
            {
                "profile_id": "sampled-v1",
                "decoding": "sampled",
                "max_new_tokens": 64,
                "seeds": [17, 29, 43],
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
                "eos_token_id": 3,
                "pad_token_id": 0,
                "batch_size": 1,
                "precision": "bf16",
                "kv_cache": True,
            },
        ],
        "bootstrap": {"confidence_level": 0.95, "resamples": 10_000, "seed": 140},
        "contamination": {
            "inventory_index_sha256": inventory_index_identity(inventories),
            "exact_method": "normalized_sha256_v1",
            "ngram_words": 8,
            "semantic_method": "minhash_lsh_review_v1",
            "scan_before_training_split": True,
        },
        "held_out_policy": {
            "decision_point": "frozen experiment plan evaluation gate",
            "opened": False,
            "opening_authorized": False,
        },
        "training_authorized": False,
        "suite_sha256": "0" * 64,
    }
    value["suite_sha256"] = suite_identity(value)
    return value


def result_manifest() -> dict[str, object]:
    return build_synthetic_result_manifest(
        repository_commit="a" * 40,
        suite_sha256="1" * 64,
    )


def rehash_suite(value: dict[str, object]) -> None:
    value["suite_sha256"] = suite_identity(value)


def rehash_result(value: dict[str, object]) -> None:
    value["resume_identity_sha256"] = resume_identity(value)
    value["result_sha256"] = result_identity(value)


def test_valid_synthetic_suite_and_result_are_non_authorizing() -> None:
    suite = suite_manifest()
    result = result_manifest()
    validate_suite_manifest(suite)
    validate_result_manifest(result)
    assert suite["training_authorized"] is False
    assert result["training_authorized"] is False


@pytest.mark.parametrize(
    "field", ["schema_id", "family_id", "model_config_sha256"]
)
def test_suite_identity_mutations_fail(field: str) -> None:
    value = suite_manifest()
    value[field] = "0" * 64
    rehash_suite(value)
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_suite_manifest(value)


def test_suite_rejects_unknown_fields_and_unhashed_mutation() -> None:
    value = suite_manifest()
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_suite_manifest(value)
    value = suite_manifest()
    value["suite_id"] = "mutated"
    with pytest.raises(ValueError, match="suite manifest identity mismatch"):
        validate_suite_manifest(value)


def test_suite_requires_complete_dimension_split_coverage() -> None:
    value = suite_manifest()
    value["inventories"].pop()
    value["contamination"]["inventory_index_sha256"] = inventory_index_identity(
        value["inventories"]
    )
    rehash_suite(value)
    with pytest.raises(ValueError, match="cover every dimension"):
        validate_suite_manifest(value)


def test_likelihood_cannot_use_generation_and_generation_binds_both_modes() -> None:
    value = suite_manifest()
    likelihood = next(
        item for item in value["inventories"] if item["dimension"] == "likelihood"
    )
    likelihood["generation_profile_ids"] = ["greedy-v1"]
    value["contamination"]["inventory_index_sha256"] = inventory_index_identity(
        value["inventories"]
    )
    rehash_suite(value)
    with pytest.raises(ValueError, match="cannot generate text"):
        validate_suite_manifest(value)

    value = suite_manifest()
    arithmetic = next(
        item for item in value["inventories"] if item["dimension"] == "arithmetic"
    )
    arithmetic["generation_profile_ids"] = ["greedy-v1"]
    value["contamination"]["inventory_index_sha256"] = inventory_index_identity(
        value["inventories"]
    )
    rehash_suite(value)
    with pytest.raises(ValueError, match="both greedy and sampled"):
        validate_suite_manifest(value)


def test_held_out_inventories_must_stay_sealed() -> None:
    value = suite_manifest()
    held_out = next(
        item for item in value["inventories"] if item["split"] == "held_out"
    )
    held_out["access_state"] = "available"
    value["contamination"]["inventory_index_sha256"] = inventory_index_identity(
        value["inventories"]
    )
    rehash_suite(value)
    with pytest.raises(ValueError, match="must be sealed"):
        validate_suite_manifest(value)

    value = suite_manifest()
    value["held_out_policy"]["opening_authorized"] = True
    rehash_suite(value)
    with pytest.raises(ValueError, match="sealed and unauthorized"):
        validate_suite_manifest(value)


def test_contamination_commitment_and_ordering_fail_closed() -> None:
    value = suite_manifest()
    value["contamination"]["inventory_index_sha256"] = "0" * 64
    rehash_suite(value)
    with pytest.raises(ValueError, match="commitment mismatch"):
        validate_suite_manifest(value)
    value = suite_manifest()
    value["contamination"]["scan_before_training_split"] = False
    rehash_suite(value)
    with pytest.raises(ValueError, match="before training split"):
        validate_suite_manifest(value)


def test_file_validation_reads_development_but_not_held_out() -> None:
    value = suite_manifest()
    validate_suite_manifest_files(value, ROOT)
    with pytest.raises(PermissionError, match="future reviewed decision"):
        validate_suite_manifest_files(value, ROOT, open_held_out=True)
    value["inventories"][0]["sha256"] = "0" * 64
    value["contamination"]["inventory_index_sha256"] = inventory_index_identity(
        value["inventories"]
    )
    rehash_suite(value)
    with pytest.raises(ValueError, match="file identity mismatch"):
        validate_suite_manifest_files(value, ROOT)


def test_result_requires_complete_separate_dimension_reports() -> None:
    value = result_manifest()
    value["dimension_reports"].pop()
    rehash_result(value)
    with pytest.raises(ValueError, match="cover every required dimension/mode"):
        validate_result_manifest(value)


def test_result_modes_and_stage_must_match_dimension_contract() -> None:
    value = result_manifest()
    likelihood = next(
        shard
        for shard in value["output_shards"]
        if shard["dimension"] == "likelihood"
    )
    likelihood["mode"] = "sampled"
    rehash_result(value)
    with pytest.raises(ValueError, match="mode does not match its dimension"):
        validate_result_manifest(value)

    value = result_manifest()
    value["output_shards"][0]["split"] = "held_out"
    rehash_result(value)
    with pytest.raises(ValueError, match="split does not match evaluation_stage"):
        validate_result_manifest(value)


def test_result_requires_every_generation_mode() -> None:
    value = result_manifest()
    value["output_shards"] = [
        shard
        for shard in value["output_shards"]
        if not (
            shard["dimension"] == "arithmetic" and shard["mode"] == "sampled"
        )
    ]
    rehash_result(value)
    with pytest.raises(ValueError, match="cover every required dimension/mode"):
        validate_result_manifest(value)


def test_held_out_result_requires_exact_authorization_binding() -> None:
    value = result_manifest()
    value["held_out_authorization"] = {
        "authorized": True,
        "decision_id": "unexpected",
        "path": "docs/unexpected.md",
        "sha256": "6" * 64,
    }
    rehash_result(value)
    with pytest.raises(ValueError, match="must not declare held-out"):
        validate_result_manifest(value)

    value = result_manifest()
    value["evaluation_stage"] = "held_out"
    for shard in value["output_shards"]:
        shard["split"] = "held_out"
    for report in value["dimension_reports"]:
        report["split"] = "held_out"
    rehash_result(value)
    with pytest.raises(ValueError, match="require explicit authorization"):
        validate_result_manifest(value)

    value["held_out_authorization"] = {
        "authorized": True,
        "decision_id": "vasu-140m-held-out-opening-v1",
        "path": "docs/VASU_140M_HELD_OUT_OPENING_DECISION.md",
        "sha256": "6" * 64,
    }
    rehash_result(value)
    validate_result_manifest(value)


def test_result_must_match_exact_suite_identity() -> None:
    suite = suite_manifest()
    value = result_manifest()
    with pytest.raises(ValueError, match="does not match suite manifest"):
        validate_result_against_suite(value, suite)
    value["suite"]["sha256"] = suite["suite_sha256"]
    rehash_result(value)
    validate_result_against_suite(value, suite)


def test_result_allows_multiple_shards_and_metrics_per_dimension() -> None:
    value = result_manifest()
    extra_shard = deepcopy(value["output_shards"][0])
    extra_shard["shard_id"] = "arithmetic-development-sampled-extra"
    extra_shard["dimension"] = "arithmetic"
    extra_shard["split"] = "development"
    extra_shard["mode"] = "sampled"
    extra_shard["path"] = (
        "evaluation/results/synthetic/arithmetic_sampled_extra.jsonl"
    )
    value["output_shards"].append(extra_shard)
    extra_report = deepcopy(value["dimension_reports"][0])
    extra_report["dimension"] = "arithmetic"
    extra_report["metric_name"] = "arithmetic_parse_rate"
    extra_report["numerator"] = 8.5
    extra_report["denominator"] = 2
    extra_report["point_estimate"] = 4.25
    extra_report["confidence_interval_95"] = [4.0, 4.5]
    extra_report["strata_path"] = (
        "evaluation/results/synthetic/arithmetic_parse_strata.json"
    )
    value["dimension_reports"].append(extra_report)
    rehash_result(value)
    validate_result_manifest(value)
    value = result_manifest()
    value["aggregate_score"] = 0.5
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_result_manifest(value)


def test_result_resume_identity_and_checkpoint_family_fail_closed() -> None:
    value = result_manifest()
    value["runtime"]["device"] = "cuda:1"
    value["result_sha256"] = result_identity(value)
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_result_manifest(value)
    value = result_manifest()
    value["checkpoint"]["family_id"] = "vasu_60m_v1"
    rehash_result(value)
    with pytest.raises(ValueError, match="checkpoint family mismatch"):
        validate_result_manifest(value)


def test_result_rejects_nonfinite_or_reversed_statistics_and_time() -> None:
    value = result_manifest()
    value["dimension_reports"][0]["point_estimate"] = float("nan")
    with pytest.raises(ValueError, match="finite number"):
        validate_result_manifest(value)
    value = result_manifest()
    value["runtime"]["completed_at"] = "2026-08-01T11:00:00+05:30"
    rehash_result(value)
    with pytest.raises(ValueError, match="precedes"):
        validate_result_manifest(value)


def test_result_mutation_without_rehash_fails() -> None:
    value = deepcopy(result_manifest())
    value["dimension_reports"][0]["metric_name"] = "changed"
    with pytest.raises(ValueError, match="result manifest identity mismatch"):
        validate_result_manifest(value)


def test_tokenizer_fixture_matches_frozen_identity() -> None:
    assert hashlib.sha256((ROOT / "assets/tokenizer.json").read_bytes()).hexdigest() == (
        TOKENIZER_SHA256
    )
