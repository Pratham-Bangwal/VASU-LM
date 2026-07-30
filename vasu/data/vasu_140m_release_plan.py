"""Validate the specification-only VASU-140M instruction seed release plan.

This module performs read-only source, review, split, deduplication, and
contamination qualification. It never writes token, mask, logical-record, or
training artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from vasu.data.instruction_quality import (
    exact_duplicate_groups,
    load_jsonl,
    load_review_decisions,
    near_duplicate_candidates,
    validate_records,
    validate_review_decisions,
)
from vasu.data.preparation.filters import comparison_normalize
from vasu.data.vasu_140m_records import (
    FAMILY_ID,
    FROZEN_FIXTURE_REPORT_SHA256,
    MODEL_CONFIG_SHA256,
    RECORD_WIDTH,
    SPECIFICATION_SHA256,
    TOKENIZER_SHA256,
    sha256_json,
)


PLAN_SCHEMA_ID = "vasu.model-family-release-plan.v1"
REPORT_SCHEMA_ID = "vasu.model-family-release-plan-report.v1"
PLAN_ID = "vasu_140m_instruction_seed_v1"
FROZEN_PLAN_SHA256 = (
    "8da8cc92ef913bb567c96ec47986f849b84eb85ac2d11f3ad245fb1639361e16"
)
FROZEN_PLAN_REPORT_SHA256 = (
    "8147215c2f58044aa60374f39e772e13fce3913090ea5c2d8acc915aeeca6fcb"
)
TRAINING_STAGE = "instruction_supervised"
ALLOCATION_ALGORITHM = (
    "preserve-source-validation-then-sha256-source-capability-largest-remainder-v1"
)
NORMALIZATION_VERSION = "vasu_cross_source_nfc_casefold_ws_v1"
NEAR_DUPLICATE_ALGORITHM = "normalized-trigram-jaccard-v1"
CONTAMINATION_ALGORITHM = "normalized-full-prompt-and-word-ngram-v1"
EXPECTED_SOURCE_IDS = (
    "vasu_instruction_quality_v1_batch_001",
    "vasu_instruction_quality_v1_batch_002",
)
EXPECTED_SPLIT_COUNTS = {
    "train": 898,
    "development": 48,
    "evaluation": 50,
}
EXPECTED_CAPABILITY_COUNTS = {
    "short_factual_qa": 247,
    "beginner_explanation": 199,
    "exact_format_following": 200,
    "rewriting_transformation": 150,
    "lists_structured_output": 100,
    "json_schema_output": 50,
    "uncertainty_honest_fallback": 50,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def sha256_file(path: Path) -> str:
    """Hash one file without modifying it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{label} fields are invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    text = _require_string(value, label)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _resolve_repository_path(
    repository_root: Path,
    value: object,
    label: str,
) -> Path:
    text = _require_string(value, label)
    pure = PurePosixPath(text.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{label} must be a safe repository-relative path")
    root = repository_root.resolve()
    resolved = root.joinpath(*pure.parts).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the repository")
    return resolved


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def load_release_plan(path: Path) -> dict[str, object]:
    """Load a strict release-plan JSON object."""

    return _load_json_object(path, "release plan")


def plan_identity(plan: Mapping[str, object]) -> str:
    """Return the canonical identity excluding the self-hash field."""

    body = dict(plan)
    body.pop("plan_sha256", None)
    return sha256_json(body)


def _validate_plan_structure(plan: Mapping[str, object]) -> None:
    _require_exact_keys(
        plan,
        {
            "schema_id",
            "plan_id",
            "plan_status",
            "created_date",
            "family",
            "accepted_contract",
            "release_scope",
            "sources",
            "split_policy",
            "deduplication_policy",
            "contamination_policy",
            "output_contract",
            "gates",
            "production_release_created",
            "training_authorized",
            "plan_sha256",
        },
        "release plan",
    )
    expected_scalars = {
        "schema_id": PLAN_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_status": "specification_only",
        "created_date": "2026-07-30",
        "production_release_created": False,
        "training_authorized": False,
    }
    for field, expected in expected_scalars.items():
        if plan[field] != expected:
            raise ValueError(f"{field} must equal {expected!r}")
    reported_identity = _require_sha256(plan["plan_sha256"], "plan_sha256")
    if plan_identity(plan) != reported_identity:
        raise ValueError("release plan hash mismatch")

    family = plan["family"]
    if not isinstance(family, Mapping):
        raise ValueError("family must be an object")
    _require_exact_keys(
        family,
        {
            "family_id",
            "model_config_sha256",
            "tokenizer_sha256",
            "record_width",
            "record_specification_sha256",
        },
        "family",
    )
    expected_family = {
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_width": RECORD_WIDTH,
        "record_specification_sha256": SPECIFICATION_SHA256,
    }
    if dict(family) != expected_family:
        raise ValueError("family identity does not match the accepted record contract")

    contract = plan["accepted_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("accepted_contract must be an object")
    _require_exact_keys(
        contract,
        {
            "fixture_report_path",
            "fixture_file_sha256",
            "fixture_report_sha256",
            "decision_path",
            "decision_sha256",
            "decision",
            "reviewer",
            "review_date",
        },
        "accepted_contract",
    )
    if contract["fixture_report_sha256"] != FROZEN_FIXTURE_REPORT_SHA256:
        raise ValueError("accepted contract fixture report identity is wrong")
    if contract["decision"] != "accepted":
        raise ValueError("accepted contract decision must be accepted")
    if contract["reviewer"] != "GPT-5.5 independent review":
        raise ValueError("accepted contract reviewer identity is wrong")
    if contract["review_date"] != "2026-07-30":
        raise ValueError("accepted contract review date is wrong")
    _require_sha256(contract["fixture_file_sha256"], "fixture_file_sha256")
    _require_sha256(contract["decision_sha256"], "decision_sha256")

    scope = plan["release_scope"]
    if not isinstance(scope, Mapping):
        raise ValueError("release_scope must be an object")
    _require_exact_keys(
        scope,
        {
            "training_stage",
            "prompt_format",
            "response_prefix",
            "language",
            "expected_source_examples",
            "quarantined_example_ids",
            "eligible_example_count",
            "capability_counts",
            "requires_compatible_base_checkpoint",
            "base_checkpoint_selected",
        },
        "release_scope",
    )
    expected_scope_scalars = {
        "training_stage": TRAINING_STAGE,
        "prompt_format": "alpaca_user_assistant_v1",
        "response_prefix": " ",
        "language": "en",
        "expected_source_examples": 1000,
        "eligible_example_count": 996,
        "requires_compatible_base_checkpoint": True,
        "base_checkpoint_selected": False,
    }
    for field, expected in expected_scope_scalars.items():
        if scope[field] != expected:
            raise ValueError(f"release_scope.{field} must equal {expected!r}")
    if scope["capability_counts"] != EXPECTED_CAPABILITY_COUNTS:
        raise ValueError("release_scope capability counts are wrong")
    quarantine = scope["quarantined_example_ids"]
    if (
        not isinstance(quarantine, list)
        or len(quarantine) != 4
        or len(set(quarantine)) != len(quarantine)
        or not all(isinstance(item, str) and item for item in quarantine)
    ):
        raise ValueError("quarantined_example_ids must contain four unique IDs")

    split = plan["split_policy"]
    if not isinstance(split, Mapping):
        raise ValueError("split_policy must be an object")
    _require_exact_keys(
        split,
        {
            "algorithm",
            "seed",
            "preserve_source_validation_as",
            "evaluation_count",
            "expected_counts",
            "assignment_sha256",
            "stratification_keys",
            "isolation_keys",
        },
        "split_policy",
    )
    if split["algorithm"] != ALLOCATION_ALGORITHM:
        raise ValueError("split allocation algorithm is wrong")
    if type(split["seed"]) is not int or split["seed"] < 0:
        raise ValueError("split seed must be a non-negative integer")
    if split["preserve_source_validation_as"] != "development":
        raise ValueError("source validation must remain development-only")
    if split["evaluation_count"] != 50:
        raise ValueError("evaluation_count must equal 50")
    if split["expected_counts"] != EXPECTED_SPLIT_COUNTS:
        raise ValueError("expected split counts are wrong")
    if split["stratification_keys"] != ["source_id", "capability"]:
        raise ValueError("split stratification keys are wrong")
    if split["isolation_keys"] != ["example_id", "normalized_semantic_sha256"]:
        raise ValueError("split isolation keys are wrong")
    _require_sha256(split["assignment_sha256"], "assignment_sha256")

    dedup = plan["deduplication_policy"]
    if not isinstance(dedup, Mapping):
        raise ValueError("deduplication_policy must be an object")
    _require_exact_keys(
        dedup,
        {
            "normalization_version",
            "exact_fields",
            "near_duplicate_algorithm",
            "near_duplicate_threshold",
            "cross_source_required",
            "cross_split_required",
        },
        "deduplication_policy",
    )
    if dedup != {
        "normalization_version": NORMALIZATION_VERSION,
        "exact_fields": [
            "instruction",
            "input",
            "response",
            "instruction_input",
            "instruction_input_response",
        ],
        "near_duplicate_algorithm": NEAR_DUPLICATE_ALGORITHM,
        "near_duplicate_threshold": 0.85,
        "cross_source_required": True,
        "cross_split_required": True,
    }:
        raise ValueError("deduplication policy is not the frozen policy")

    contamination = plan["contamination_policy"]
    if not isinstance(contamination, Mapping):
        raise ValueError("contamination_policy must be an object")
    _require_exact_keys(
        contamination,
        {
            "algorithm",
            "normalization_version",
            "ngram_words",
            "prompt_inventories",
            "expected_prompt_count",
            "expected_exact_matches",
            "expected_ngram_only_matches",
            "eligible_exact_matches",
            "eligible_ngram_only_matches",
        },
        "contamination_policy",
    )
    if contamination["algorithm"] != CONTAMINATION_ALGORITHM:
        raise ValueError("contamination algorithm is wrong")
    if contamination["normalization_version"] != NORMALIZATION_VERSION:
        raise ValueError("contamination normalization version is wrong")
    if contamination["ngram_words"] != 8:
        raise ValueError("contamination ngram_words must equal 8")
    if contamination["expected_prompt_count"] != 2618:
        raise ValueError("expected_prompt_count must equal 2618")
    if contamination["expected_ngram_only_matches"] != 0:
        raise ValueError("expected_ngram_only_matches must equal zero")
    if contamination["eligible_exact_matches"] != 0:
        raise ValueError("eligible_exact_matches must equal zero")
    if contamination["eligible_ngram_only_matches"] != 0:
        raise ValueError("eligible_ngram_only_matches must equal zero")
    expected_matches = contamination["expected_exact_matches"]
    if not isinstance(expected_matches, list) or len(expected_matches) != 4:
        raise ValueError("expected_exact_matches must contain four findings")
    inventories = contamination["prompt_inventories"]
    if not isinstance(inventories, list) or len(inventories) != 10:
        raise ValueError("prompt_inventories must contain ten pinned files")

    output = plan["output_contract"]
    if not isinstance(output, Mapping):
        raise ValueError("output_contract must be an object")
    _require_exact_keys(
        output,
        {
            "release_directory",
            "logical_manifest_path",
            "train_token_path",
            "train_mask_path",
            "development_token_path",
            "development_mask_path",
            "evaluation_token_path",
            "evaluation_mask_path",
            "overwrite_allowed",
        },
        "output_contract",
    )
    if output["overwrite_allowed"] is not False:
        raise ValueError("release output overwrite must be forbidden")

    gates = plan["gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("gates must be an object")
    expected_gates = {
        "record_contract_accepted": True,
        "source_reviews_complete": True,
        "combined_deduplication_passed": True,
        "contamination_quarantine_specified": True,
        "production_release_review_accepted": False,
        "compatible_base_checkpoint_selected": False,
        "training_plan_accepted": False,
        "training_authorized": False,
    }
    if dict(gates) != expected_gates:
        raise ValueError("release gates do not match specification-only status")
    if reported_identity != FROZEN_PLAN_SHA256:
        raise ValueError("release plan does not match the frozen plan identity")


def _verify_bound_file(
    repository_root: Path,
    path_value: object,
    hash_value: object,
    label: str,
) -> Path:
    path = _resolve_repository_path(repository_root, path_value, f"{label}.path")
    expected = _require_sha256(hash_value, f"{label}.sha256")
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    if sha256_file(path) != expected:
        raise ValueError(f"{label} SHA-256 mismatch")
    return path


def _collect_prompt_items(
    value: object,
    *,
    benchmark_path: str,
) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        prompt_id = value.get("id")
        prompt = value.get("prompt")
        if isinstance(prompt_id, str) and isinstance(prompt, str):
            prompts.append(
                {
                    "benchmark_path": benchmark_path,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                }
            )
        for child in value.values():
            prompts.extend(
                _collect_prompt_items(child, benchmark_path=benchmark_path)
            )
    elif isinstance(value, list):
        for child in value:
            prompts.extend(
                _collect_prompt_items(child, benchmark_path=benchmark_path)
            )
    return prompts


def _contamination_findings(
    records: Sequence[dict[str, Any]],
    prompts: Sequence[dict[str, str]],
    *,
    ngram_words: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    prepared_prompts: list[tuple[dict[str, str], str, tuple[str, ...]]] = []
    for prompt in prompts:
        normalized = comparison_normalize(prompt["prompt"])
        words = _WORD_RE.findall(normalized)
        fragments = tuple(
            " ".join(words[index : index + ngram_words])
            for index in range(max(0, len(words) - ngram_words + 1))
        )
        prepared_prompts.append((prompt, normalized, fragments))

    exact: list[dict[str, str]] = []
    ngram_only: list[dict[str, str]] = []
    for record in records:
        text = comparison_normalize(
            " ".join(
                str(record.get(field, ""))
                for field in ("instruction", "input", "response")
            )
        )
        for prompt, normalized, fragments in prepared_prompts:
            finding = {
                "example_id": str(record["example_id"]),
                "benchmark_path": prompt["benchmark_path"],
                "prompt_id": prompt["prompt_id"],
            }
            if normalized and normalized in text:
                exact.append({**finding, "matching_method": "full_prompt"})
            elif fragments and any(fragment in text for fragment in fragments):
                ngram_only.append(
                    {**finding, "matching_method": f"word_{ngram_words}_gram"}
                )
    def key(item: dict[str, str]) -> tuple[str, str, str, str]:
        return (
            item["example_id"],
            item["benchmark_path"],
            item["prompt_id"],
            item["matching_method"],
        )

    return sorted(exact, key=key), sorted(ngram_only, key=key)


def _largest_remainder_quotas(
    sizes: Mapping[tuple[str, str], int],
    total: int,
) -> dict[tuple[str, str], int]:
    population = sum(sizes.values())
    if not 0 < total < population:
        raise ValueError("held-out count must be within the eligible population")
    exact = {
        key: sizes[key] * total / population
        for key in sizes
    }
    quotas = {key: int(exact[key]) for key in sizes}
    remaining = total - sum(quotas.values())
    ranked = sorted(
        sizes,
        key=lambda key: (-(exact[key] - quotas[key]), key),
    )
    for key in ranked[:remaining]:
        quotas[key] += 1
    return quotas


def _derive_assignments(
    records: Sequence[dict[str, Any]],
    source_ids: Mapping[str, str],
    source_splits: Mapping[str, str],
    *,
    seed: int,
    evaluation_count: int,
) -> dict[str, str]:
    assignments = {
        example_id: "development"
        for example_id, split in source_splits.items()
        if split == "validation"
    }
    train_records = [
        record
        for record in records
        if source_splits[str(record["example_id"])] == "train"
    ]
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in train_records:
        example_id = str(record["example_id"])
        key = (source_ids[example_id], str(record["capability"]))
        strata[key].append(record)
    quotas = _largest_remainder_quotas(
        {key: len(rows) for key, rows in strata.items()},
        evaluation_count,
    )
    for key, rows in strata.items():
        ordered = sorted(
            rows,
            key=lambda record: hashlib.sha256(
                (
                    f"{seed}:{key[0]}:{key[1]}:"
                    f"{record['example_id']}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        held_out = {
            str(record["example_id"])
            for record in ordered[: quotas[key]]
        }
        for record in ordered:
            example_id = str(record["example_id"])
            assignments[example_id] = (
                "evaluation" if example_id in held_out else "train"
            )
    return assignments


def _assignment_identity(
    assignments: Mapping[str, str],
    source_ids: Mapping[str, str],
) -> str:
    rows = [
        {
            "example_id": example_id,
            "source_id": source_ids[example_id],
            "split": assignments[example_id],
        }
        for example_id in sorted(assignments)
    ]
    return sha256_json(rows)


def validate_release_plan(
    plan: Mapping[str, object],
    repository_root: Path,
    *,
    require_outputs_absent: bool = True,
) -> dict[str, object]:
    """Validate the plan and return deterministic, read-only qualification evidence."""

    _validate_plan_structure(plan)
    root = repository_root.resolve()
    contract = plan["accepted_contract"]
    assert isinstance(contract, Mapping)
    _verify_bound_file(
        root,
        contract["fixture_report_path"],
        contract["fixture_file_sha256"],
        "accepted fixture",
    )
    _verify_bound_file(
        root,
        contract["decision_path"],
        contract["decision_sha256"],
        "accepted decision",
    )

    sources = plan["sources"]
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("sources must contain exactly two reviewed batches")
    source_records: list[dict[str, Any]] = []
    source_ids_by_example: dict[str, str] = {}
    source_splits: dict[str, str] = {}
    source_evidence: list[dict[str, object]] = []
    observed_source_ids: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"sources[{index}] must be an object")
        _require_exact_keys(
            source,
            {
                "source_id",
                "manifest_path",
                "manifest_sha256",
                "source_path",
                "source_sha256",
                "review_path",
                "review_sha256",
                "approved_example_count",
                "license",
                "provenance",
            },
            f"sources[{index}]",
        )
        source_id = _require_string(source["source_id"], "source_id")
        observed_source_ids.append(source_id)
        manifest_path = _verify_bound_file(
            root,
            source["manifest_path"],
            source["manifest_sha256"],
            f"{source_id} manifest",
        )
        source_path = _verify_bound_file(
            root,
            source["source_path"],
            source["source_sha256"],
            f"{source_id} source",
        )
        review_path = _verify_bound_file(
            root,
            source["review_path"],
            source["review_sha256"],
            f"{source_id} review",
        )
        manifest = _load_json_object(manifest_path, f"{source_id} manifest")
        if manifest.get("training_authorized") is not False:
            raise ValueError(f"{source_id} manifest authorizes training")
        if manifest.get("tokenizer_sha256") != TOKENIZER_SHA256:
            raise ValueError(f"{source_id} tokenizer identity mismatch")
        if manifest.get("source_sha256") != source["source_sha256"]:
            raise ValueError(f"{source_id} source identity mismatch")
        if manifest.get("review_sha256") != source["review_sha256"]:
            raise ValueError(f"{source_id} review identity mismatch")
        if manifest.get("truncated_examples") != 0:
            raise ValueError(f"{source_id} contains truncated examples")
        if manifest.get("cross_split_leakage") != []:
            raise ValueError(f"{source_id} reports cross-split leakage")
        records = load_jsonl(source_path)
        decisions = load_review_decisions(review_path)
        findings = validate_records(records) + validate_review_decisions(
            records, decisions
        )
        if findings:
            finding = findings[0]
            raise ValueError(
                f"{source_id} validation failed: {finding.code}: {finding.message}"
            )
        if len(records) != source["approved_example_count"]:
            raise ValueError(f"{source_id} approved count mismatch")
        if set(decisions) != {str(record["example_id"]) for record in records}:
            raise ValueError(f"{source_id} review does not cover every record")
        if any(decision["status"] != "approved" for decision in decisions.values()):
            raise ValueError(f"{source_id} contains a non-approved decision")
        if set(manifest.get("approved_example_ids", [])) != set(decisions):
            raise ValueError(f"{source_id} manifest approved IDs mismatch")
        split_assignments = manifest.get("split_assignments")
        if (
            not isinstance(split_assignments, dict)
            or set(split_assignments) != set(decisions)
            or set(split_assignments.values()) != {"train", "validation"}
        ):
            raise ValueError(f"{source_id} source split assignments are invalid")
        expected_license = _require_string(source["license"], "source license")
        expected_provenance = _require_string(
            source["provenance"], "source provenance"
        )
        for record in records:
            example_id = str(record["example_id"])
            if example_id in source_ids_by_example:
                raise ValueError(f"duplicate example ID across sources: {example_id}")
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"{source_id} record metadata is invalid")
            if metadata.get("license") != expected_license:
                raise ValueError(f"{source_id} record license mismatch")
            if metadata.get("provenance") != expected_provenance:
                raise ValueError(f"{source_id} record provenance mismatch")
            source_ids_by_example[example_id] = source_id
            source_splits[example_id] = split_assignments[example_id]
        source_records.extend(records)
        source_evidence.append(
            {
                "source_id": source_id,
                "record_count": len(records),
                "manifest_sha256": source["manifest_sha256"],
                "source_sha256": source["source_sha256"],
                "review_sha256": source["review_sha256"],
                "license": expected_license,
                "all_human_approved": True,
            }
        )
    if tuple(observed_source_ids) != EXPECTED_SOURCE_IDS:
        raise ValueError("source IDs or source order do not match the frozen plan")

    scope = plan["release_scope"]
    assert isinstance(scope, Mapping)
    quarantine = set(scope["quarantined_example_ids"])
    all_ids = {str(record["example_id"]) for record in source_records}
    if not quarantine <= all_ids:
        raise ValueError("quarantine contains an unknown example ID")
    eligible = [
        record
        for record in source_records
        if str(record["example_id"]) not in quarantine
    ]
    if len(source_records) != scope["expected_source_examples"]:
        raise ValueError("source example count does not match the release scope")
    if len(eligible) != scope["eligible_example_count"]:
        raise ValueError("eligible example count does not match the release scope")
    if exact_duplicate_groups(source_records):
        raise ValueError("combined sources contain exact duplicates")
    threshold = float(plan["deduplication_policy"]["near_duplicate_threshold"])
    if near_duplicate_candidates(source_records, threshold):
        raise ValueError("combined sources contain high-confidence near duplicates")
    capability_counts = dict(
        Counter(str(record["capability"]) for record in eligible)
    )
    if capability_counts != EXPECTED_CAPABILITY_COUNTS:
        raise ValueError("eligible capability counts do not match the plan")
    semantic_hashes = [
        sha256_json(
            {
                "instruction": comparison_normalize(str(record["instruction"])),
                "input": comparison_normalize(str(record["input"])),
                "response": comparison_normalize(str(record["response"])),
            }
        )
        for record in eligible
    ]
    if len(set(semantic_hashes)) != len(semantic_hashes):
        raise ValueError("eligible records contain semantic-hash collisions")

    contamination = plan["contamination_policy"]
    assert isinstance(contamination, Mapping)
    prompts: list[dict[str, str]] = []
    inventory_evidence: list[dict[str, object]] = []
    for index, inventory in enumerate(contamination["prompt_inventories"]):
        if not isinstance(inventory, Mapping):
            raise ValueError(f"prompt inventory {index} must be an object")
        _require_exact_keys(
            inventory,
            {"path", "sha256"},
            f"prompt inventory {index}",
        )
        inventory_path = _verify_bound_file(
            root,
            inventory["path"],
            inventory["sha256"],
            f"prompt inventory {index}",
        )
        relative = str(inventory["path"]).replace("\\", "/")
        if inventory_path.suffix == ".jsonl":
            items = []
            for line_number, line in enumerate(
                inventory_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"prompt inventory {index} has invalid JSONL at "
                        f"line {line_number}"
                    ) from error
                items.extend(
                    _collect_prompt_items(payload, benchmark_path=relative)
                )
        else:
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            items = _collect_prompt_items(payload, benchmark_path=relative)
        prompts.extend(items)
        inventory_evidence.append(
            {
                "path": relative,
                "sha256": inventory["sha256"],
                "prompt_count": len(items),
            }
        )
    if len(prompts) != contamination["expected_prompt_count"]:
        raise ValueError("prompt inventory count mismatch")
    exact, ngram_only = _contamination_findings(
        source_records,
        prompts,
        ngram_words=int(contamination["ngram_words"]),
    )
    if exact != contamination["expected_exact_matches"]:
        raise ValueError("exact contamination findings do not match the plan")
    if len(ngram_only) != contamination["expected_ngram_only_matches"]:
        raise ValueError("ngram-only contamination findings do not match the plan")
    eligible_exact, eligible_ngram = _contamination_findings(
        eligible,
        prompts,
        ngram_words=int(contamination["ngram_words"]),
    )
    if len(eligible_exact) != contamination["eligible_exact_matches"]:
        raise ValueError("eligible exact contamination is not zero")
    if len(eligible_ngram) != contamination["eligible_ngram_only_matches"]:
        raise ValueError("eligible ngram contamination is not zero")

    split = plan["split_policy"]
    assert isinstance(split, Mapping)
    eligible_ids = {str(record["example_id"]) for record in eligible}
    assignments = _derive_assignments(
        eligible,
        {key: value for key, value in source_ids_by_example.items() if key in eligible_ids},
        {key: value for key, value in source_splits.items() if key in eligible_ids},
        seed=int(split["seed"]),
        evaluation_count=int(split["evaluation_count"]),
    )
    counts = dict(Counter(assignments.values()))
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError("derived split counts do not match the plan")
    assignment_sha256 = _assignment_identity(assignments, source_ids_by_example)
    if assignment_sha256 != split["assignment_sha256"]:
        raise ValueError("derived split assignment identity mismatch")
    if any(
        assignments[example_id] != "development"
        for example_id in eligible_ids
        if source_splits[example_id] == "validation"
    ):
        raise ValueError("source validation record escaped development split")

    output = plan["output_contract"]
    assert isinstance(output, Mapping)
    output_paths = {
        field: str(value).replace("\\", "/")
        for field, value in output.items()
        if field != "overwrite_allowed"
    }
    for field, value in output_paths.items():
        path = _resolve_repository_path(root, value, f"output_contract.{field}")
        if require_outputs_absent and path.exists():
            raise ValueError(f"planned production output already exists: {value}")

    report_body: dict[str, object] = {
        "schema_id": REPORT_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_sha256": plan["plan_sha256"],
        "specification_only": True,
        "source_evidence": source_evidence,
        "source_example_count": len(source_records),
        "quarantined_example_ids": sorted(quarantine),
        "eligible_example_count": len(eligible),
        "capability_counts": capability_counts,
        "combined_exact_duplicate_groups": 0,
        "combined_near_duplicate_candidates": 0,
        "prompt_inventories": inventory_evidence,
        "prompt_count": len(prompts),
        "exact_contamination_findings": exact,
        "ngram_only_contamination_findings": ngram_only,
        "eligible_contamination_findings": 0,
        "split_counts": counts,
        "assignment_sha256": assignment_sha256,
        "planned_output_paths": output_paths,
        "planned_outputs_absent": True,
        "production_release_created": False,
        "release_build_permitted": False,
        "training_authorized": False,
        "training_permitted": False,
    }
    report_body["report_sha256"] = sha256_json(report_body)
    return report_body


def validate_plan_report(report: Mapping[str, object]) -> None:
    """Validate report structure and canonical self-consistency."""

    _require_exact_keys(
        report,
        {
            "schema_id",
            "plan_id",
            "plan_sha256",
            "specification_only",
            "source_evidence",
            "source_example_count",
            "quarantined_example_ids",
            "eligible_example_count",
            "capability_counts",
            "combined_exact_duplicate_groups",
            "combined_near_duplicate_candidates",
            "prompt_inventories",
            "prompt_count",
            "exact_contamination_findings",
            "ngram_only_contamination_findings",
            "eligible_contamination_findings",
            "split_counts",
            "assignment_sha256",
            "planned_output_paths",
            "planned_outputs_absent",
            "production_release_created",
            "release_build_permitted",
            "training_authorized",
            "training_permitted",
            "report_sha256",
        },
        "release plan report",
    )
    expected_scalars = {
        "schema_id": REPORT_SCHEMA_ID,
        "plan_id": PLAN_ID,
        "plan_sha256": FROZEN_PLAN_SHA256,
        "specification_only": True,
        "source_example_count": 1000,
        "eligible_example_count": 996,
        "combined_exact_duplicate_groups": 0,
        "combined_near_duplicate_candidates": 0,
        "prompt_count": 2618,
        "eligible_contamination_findings": 0,
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "assignment_sha256": (
            "59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394"
        ),
        "planned_outputs_absent": True,
        "production_release_created": False,
        "release_build_permitted": False,
        "training_authorized": False,
        "training_permitted": False,
    }
    for field, expected in expected_scalars.items():
        if report[field] != expected:
            raise ValueError(f"release plan report {field} is invalid")
    reported_hash = _require_sha256(report["report_sha256"], "report_sha256")
    body = dict(report)
    del body["report_sha256"]
    if sha256_json(body) != reported_hash:
        raise ValueError("release plan report hash mismatch")


def validate_frozen_plan_report(report: Mapping[str, object]) -> None:
    """Validate the exact checked-in release-plan evidence identity."""

    validate_plan_report(report)
    if report["report_sha256"] != FROZEN_PLAN_REPORT_SHA256:
        raise ValueError("release plan report does not match the frozen identity")
