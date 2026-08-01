"""Immutable, non-authorizing construction plan for VASU-140M eval v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from evaluation.framework.vasu_140m_base_v2 import (
    DIMENSIONS,
    INTERFACES,
    TOKENIZER_SHA256,
)


SCHEMA_ID = "vasu_140m_base_evaluation_inventory_construction_plan_v1"
SPLITS = frozenset({"development", "held_out"})
PROMPT_DIMENSIONS = DIMENSIONS - {"likelihood"}
COUNTS = {
    "likelihood": {"development": 512, "held_out": 512},
    "factuality": {"development": 200, "held_out": 200},
    "arithmetic": {"development": 1_000, "held_out": 1_000},
    "repetition": {"development": 120, "held_out": 120},
    "robustness": {"development": 120, "held_out": 120},
    "manual_review": {"development": 60, "held_out": 60},
}
MODES = {
    "likelihood": "post_acquisition_document_holdout",
    "factuality": "source_attributed_independent_authoring",
    "arithmetic": "deterministic_verified_generation",
    "repetition": "independent_raw_continuation_authoring",
    "robustness": "independent_paired_perturbation_authoring",
    "manual_review": "independent_stratified_raw_continuation_authoring",
}
SOURCE_USAGE_POLICIES = {
    "development_seed_only": (True, False),
    "deterministic_generator": (True, True),
    "topic_seed_only": (False, False),
}
EXPECTED_SOURCE_IDS = {
    "likelihood": [],
    "factuality": ["factual-cpt-v2-development-seed"],
    "arithmetic": ["verified-arithmetic-v2-generator"],
    "repetition": ["ultrachat-promotion-topic-seed"],
    "robustness": ["ultrachat-promotion-topic-seed"],
    "manual_review": ["ultrachat-promotion-topic-seed"],
}
DEPENDENCY_KEYS = frozenset(
    {
        "scoring_qualification_decision",
        "scoring_postcommit_identity_decision",
        "inventory_contract_decision",
        "inventory_postcommit_identity_decision",
    }
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def plan_identity(plan: Mapping[str, object]) -> str:
    body = dict(plan)
    body.pop("plan_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


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


def _url(value: object, label: str) -> str:
    text = _string(value, label)
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return text


def _commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return text


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return text


def _binding(value: object, label: str) -> tuple[str, str]:
    item = _mapping(value, label)
    _exact(item, {"path", "sha256"}, label)
    return _safe_path(item["path"], f"{label}.path"), _sha(
        item["sha256"], f"{label}.sha256"
    )


def validate_inventory_construction_plan(plan: Mapping[str, object]) -> None:
    """Validate the frozen construction decision without authorizing content."""

    _exact(
        plan,
        {
            "schema_id",
            "plan_id",
            "repository_commit",
            "suite_id",
            "tokenizer",
            "implementation",
            "dependencies",
            "source_catalog",
            "dimensions",
            "split_policy",
            "contamination_policy",
            "held_out_security",
            "review",
            "construction_authorized",
            "evaluation_run_authorized",
            "training_authorized",
            "plan_sha256",
        },
        "inventory construction plan",
    )
    if plan["schema_id"] != SCHEMA_ID:
        raise ValueError("inventory construction plan schema mismatch")
    _string(plan["plan_id"], "plan_id")
    _commit(plan["repository_commit"], "repository_commit")
    _string(plan["suite_id"], "suite_id")
    _, tokenizer_sha = _binding(plan["tokenizer"], "tokenizer")
    if tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError("inventory plan tokenizer identity mismatch")
    implementation = _mapping(plan["implementation"], "implementation")
    _exact(implementation, {"inventory_contract", "task_scorers"}, "implementation")
    _binding(implementation["inventory_contract"], "implementation.inventory_contract")
    _binding(implementation["task_scorers"], "implementation.task_scorers")
    dependencies = _mapping(plan["dependencies"], "dependencies")
    if set(dependencies) != DEPENDENCY_KEYS:
        raise ValueError("construction plan dependencies are incomplete")
    for name, value in dependencies.items():
        path, _ = _binding(value, f"dependencies.{name}")
        if not path.startswith("docs/") or not path.endswith(".md"):
            raise ValueError("construction plan dependency must be a decision document")

    catalog = plan["source_catalog"]
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("source_catalog must be a non-empty list")
    source_ids: set[str] = set()
    for index, raw in enumerate(catalog):
        label = f"source catalog entry {index}"
        source = _mapping(raw, label)
        _exact(
            source,
            {
                "source_id",
                "artifact",
                "role",
                "usage_policy",
                "provenance",
                "license",
                "prompt_reuse_permitted",
                "held_out_derivation_permitted",
            },
            label,
        )
        source_id = _string(source["source_id"], f"{label}.source_id")
        if source_id in source_ids:
            raise ValueError("source catalog IDs must be unique")
        source_ids.add(source_id)
        _binding(source["artifact"], f"{label}.artifact")
        _string(source["role"], f"{label}.role")
        usage_policy = _string(source["usage_policy"], f"{label}.usage_policy")
        if usage_policy not in SOURCE_USAGE_POLICIES:
            raise ValueError(f"{label}.usage_policy is unsupported")
        _string(source["provenance"], f"{label}.provenance")
        license_value = _mapping(source["license"], f"{label}.license")
        _exact(license_value, {"name", "url"}, f"{label}.license")
        _string(license_value["name"], f"{label}.license.name")
        _url(license_value["url"], f"{label}.license.url")
        for field in ("prompt_reuse_permitted", "held_out_derivation_permitted"):
            if not isinstance(source[field], bool):
                raise ValueError(f"{label}.{field} must be boolean")
        expected_reuse = SOURCE_USAGE_POLICIES[usage_policy]
        if (
            source["prompt_reuse_permitted"],
            source["held_out_derivation_permitted"],
        ) != expected_reuse:
            raise ValueError(f"{label} reuse permissions do not match usage_policy")

    dimensions = _mapping(plan["dimensions"], "dimensions")
    if set(dimensions) != DIMENSIONS:
        raise ValueError("dimension plans must cover all six dimensions")
    for dimension, raw in dimensions.items():
        label = f"dimension {dimension}"
        item = _mapping(raw, label)
        _exact(
            item,
            {
                "interface",
                "construction_mode",
                "counts",
                "count_scope",
                "source_ids",
                "selection_unit",
                "family_isolation",
                "provenance_required",
                "contamination_commitment_required",
                "scorer_id",
            },
            label,
        )
        if item["interface"] != INTERFACES[dimension]:
            raise ValueError(f"{label} interface mismatch")
        if item["construction_mode"] != MODES[dimension]:
            raise ValueError(f"{label} construction mode mismatch")
        counts = _mapping(item["counts"], f"{label}.counts")
        if set(counts) != SPLITS or dict(counts) != COUNTS[dimension]:
            raise ValueError(f"{label} frozen counts mismatch")
        expected_scope = "per_admitted_source" if dimension == "likelihood" else "total"
        if item["count_scope"] != expected_scope:
            raise ValueError(f"{label} count scope mismatch")
        ids = item["source_ids"]
        if not isinstance(ids, list):
            raise ValueError(f"{label}.source_ids must be a list")
        if ids != EXPECTED_SOURCE_IDS[dimension]:
            raise ValueError(f"{label}.source_ids do not match the frozen assignment")
        if dimension != "likelihood" and (not ids or set(ids) - source_ids):
            raise ValueError(f"{label}.source_ids are missing or unknown")
        _string(item["selection_unit"], f"{label}.selection_unit")
        _string(item["scorer_id"], f"{label}.scorer_id")
        for field in (
            "family_isolation",
            "provenance_required",
            "contamination_commitment_required",
        ):
            if item[field] is not True:
                raise ValueError(f"{label}.{field} must be true")

    split = _mapping(plan["split_policy"], "split_policy")
    _exact(
        split,
        {
            "seed",
            "algorithm",
            "semantic_family_isolation",
            "parent_document_isolation",
            "development_before_held_out",
            "no_cross_split_derivation",
        },
        "split_policy",
    )
    _positive_int(split["seed"], "split_policy.seed")
    if split["algorithm"] != "sha256_rank_by_family_v1":
        raise ValueError("split assignment algorithm mismatch")
    for field in (
        "semantic_family_isolation",
        "parent_document_isolation",
        "development_before_held_out",
        "no_cross_split_derivation",
    ):
        if split[field] is not True:
            raise ValueError(f"split_policy.{field} must be true")

    contamination = _mapping(plan["contamination_policy"], "contamination_policy")
    _exact(
        contamination,
        {
            "exact_prompt_hash",
            "answer_hash",
            "fragment_words",
            "semantic_review",
            "scan_before_data_split",
        },
        "contamination_policy",
    )
    if contamination != {
        "exact_prompt_hash": "sha256_nfc_casefold_whitespace_v1",
        "answer_hash": "sha256_nfc_casefold_whitespace_v1",
        "fragment_words": 8,
        "semantic_review": "minhash_lsh_plus_human_v1",
        "scan_before_data_split": True,
    }:
        raise ValueError("contamination policy does not match the frozen contract")

    security = _mapping(plan["held_out_security"], "held_out_security")
    _exact(
        security,
        {
            "algorithm",
            "recipient_fingerprint",
            "plaintext_repository_path_forbidden",
            "independent_curator_required",
            "opening_authorized",
            "key_creation_authorized",
        },
        "held_out_security",
    )
    if (
        security["algorithm"] != "age-x25519"
        or security["recipient_fingerprint"] is not None
        or security["plaintext_repository_path_forbidden"] is not True
        or security["independent_curator_required"] is not True
        or security["opening_authorized"] is not False
        or security["key_creation_authorized"] is not False
    ):
        raise ValueError("held-out security must remain unkeyed and non-authorizing")

    review = _mapping(plan["review"], "review")
    _exact(review, {"status", "decision_path", "decision_sha256"}, "review")
    if review != {
        "status": "pending",
        "decision_path": None,
        "decision_sha256": None,
    }:
        raise ValueError("inventory construction plan review must be pending")
    for field in (
        "construction_authorized",
        "evaluation_run_authorized",
        "training_authorized",
    ):
        if plan[field] is not False:
            raise ValueError(f"{field} must be false")
    if _sha(plan["plan_sha256"], "plan_sha256") != plan_identity(plan):
        raise ValueError("inventory construction plan identity mismatch")


def validate_inventory_construction_plan_files(
    plan: Mapping[str, object], repository_root: Path
) -> None:
    """Verify every currently available implementation/source artifact."""

    validate_inventory_construction_plan(plan)
    root = repository_root.resolve()
    bindings: list[tuple[str, str]] = []
    tokenizer = _mapping(plan["tokenizer"], "tokenizer")
    bindings.append(_binding(tokenizer, "tokenizer"))
    implementation = _mapping(plan["implementation"], "implementation")
    for name, value in implementation.items():
        bindings.append(_binding(value, f"implementation.{name}"))
    dependencies = _mapping(plan["dependencies"], "dependencies")
    for name, value in dependencies.items():
        bindings.append(_binding(value, f"dependencies.{name}"))
    for index, source in enumerate(plan["source_catalog"]):
        bindings.append(_binding(source["artifact"], f"source catalog entry {index}.artifact"))
    seen: set[str] = set()
    for relative, expected in bindings:
        if relative.casefold() in seen:
            raise ValueError("construction plan artifact paths must be unique")
        seen.add(relative.casefold())
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"construction plan artifact is missing: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"construction plan artifact identity mismatch: {relative}")
