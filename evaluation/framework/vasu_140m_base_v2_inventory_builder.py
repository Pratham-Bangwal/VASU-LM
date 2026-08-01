"""Fixture-only builder for VASU-140M evaluation-v2 development inventories.

Production prompt construction remains gated by the independently reviewed
construction plan.  This module qualifies deterministic payload, provenance,
contamination, and manifest assembly without exposing a production mode.
"""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from evaluation.framework.vasu_140m_base_v2 import (
    INTERFACES,
    RESULT_MODES,
    TOKENIZER_SHA256,
    canonical_json,
    sha256_file,
)
from evaluation.framework.vasu_140m_base_v2_inventory import (
    CONTAMINATION_RECORD_SCHEMA_ID,
    INVENTORY_SCHEMA_ID,
    PAYLOAD_RECORD_SCHEMA_ID,
    PROVENANCE_RECORD_SCHEMA_ID,
    content_text_sha256,
    inventory_identity,
    normalized_text_sha256,
    normalized_word_count,
    provenance_identity,
    validate_contamination_record,
    validate_inventory_manifest,
    validate_inventory_manifest_files,
    validate_payload_record,
    validate_provenance_record,
)
from evaluation.framework.vasu_140m_base_v2_inventory_plan import (
    plan_identity,
    validate_inventory_construction_plan,
)
from evaluation.framework.vasu_140m_base_v2_tasks import (
    TASK_SCHEMA_ID,
    task_identity,
    validate_inventory,
    validate_task,
)


AUTHORING_SCHEMA_ID = "vasu_140m_base_evaluation_development_authoring_item_v1"
BUILDER_REPORT_SCHEMA_ID = "vasu_140m_base_evaluation_inventory_builder_report_v1"
SUPPORTED_DIMENSIONS = frozenset(
    {"factuality", "arithmetic", "repetition", "robustness", "manual_review"}
)
AUTHORING_FIELDS = frozenset(
    {
        "schema_id",
        "item_id",
        "dimension",
        "semantic_family_id",
        "parent_document_id",
        "strata",
        "content",
        "scoring",
        "provenance",
        "human_approved",
    }
)
PROVENANCE_INPUT_FIELDS = frozenset(
    {
        "source_name",
        "source_url",
        "license_name",
        "license_url",
        "source_revision",
        "citation",
        "retrieved_at",
        "human_authored",
    }
)


def _exact(value: Mapping[str, object], expected: set[str] | frozenset[str], label: str) -> None:
    missing = sorted(set(expected) - set(value))
    unknown = sorted(set(value) - set(expected))
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


def _commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase Git commit")
    return text


def _safe_relative(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"{label} must be a safe repository-relative path")
    return text


def _is_junction(path: Path) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _reject_link_components(path: Path, root: Path, label: str) -> None:
    current = path
    while current != root:
        if current.exists() and (current.is_symlink() or _is_junction(current)):
            raise ValueError(f"{label} may not traverse a link or junction")
        current = current.parent


def _normalized_words(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", " ".join(value.strip().split()))
    return normalized.split()


def _ngram_hashes(values: Sequence[str], width: int = 8) -> list[str]:
    hashes: set[str] = set()
    for value in values:
        words = _normalized_words(value)
        for index in range(len(words) - width + 1):
            fragment = " ".join(words[index : index + width])
            hashes.add(normalized_text_sha256(fragment))
    return sorted(hashes)


def _exact_commitments(values: Sequence[str]) -> list[dict[str, object]]:
    commitments = {
        normalized_text_sha256(value): normalized_word_count(value) for value in values
    }
    return [
        {"sha256": digest, "word_count": commitments[digest]}
        for digest in sorted(commitments)
    ]


def _prompt_values(dimension: str, content: Mapping[str, object]) -> list[str]:
    if dimension == "robustness":
        return [
            _string(content["baseline_prompt"], "baseline_prompt"),
            _string(content["variant_prompt"], "variant_prompt"),
        ]
    return [_string(content["prompt"], "prompt")]


def _answer_values(
    dimension: str,
    content: Mapping[str, object],
    scoring: Mapping[str, object],
) -> list[str]:
    if dimension == "factuality":
        correct = scoring.get("correct_choice_id")
        choices = content.get("choices")
        if not isinstance(choices, list):
            return []
        return [
            str(choice["text"])
            for choice in choices
            if isinstance(choice, Mapping) and choice.get("choice_id") == correct
        ]
    if dimension == "arithmetic":
        return [str(scoring.get("expected_answer", ""))]
    if dimension == "robustness":
        answers = scoring.get("accepted_answers")
        return list(answers) if isinstance(answers, list) else []
    return []


def _build_content_and_input(
    dimension: str,
    content: Mapping[str, object],
    item_id: str,
    strata: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if dimension == "factuality":
        _exact(content, {"prompt", "choices"}, "factuality content")
        prompt = _string(content["prompt"], "content.prompt")
        choices = content["choices"]
        if not isinstance(choices, list):
            raise ValueError("factuality choices must be a list")
        payload_content = {
            "prompt": prompt,
            "prompt_sha256": content_text_sha256(prompt),
            "choices": [dict(_mapping(choice, "choice")) for choice in choices],
        }
        task_input = {
            "prompt_sha256": payload_content["prompt_sha256"],
            "choice_ids": [str(choice["choice_id"]) for choice in payload_content["choices"]],
        }
    elif dimension in {"arithmetic", "repetition", "manual_review"}:
        _exact(content, {"prompt"}, f"{dimension} content")
        prompt = _string(content["prompt"], "content.prompt")
        payload_content = {
            "prompt": prompt,
            "prompt_sha256": content_text_sha256(prompt),
        }
        task_input = {"prompt_sha256": payload_content["prompt_sha256"]}
    else:
        _exact(
            content,
            {"baseline_prompt", "variant_prompt"},
            "robustness content",
        )
        baseline = _string(content["baseline_prompt"], "baseline_prompt")
        variant = _string(content["variant_prompt"], "variant_prompt")
        payload_content = {
            "baseline_prompt": baseline,
            "baseline_prompt_sha256": content_text_sha256(baseline),
            "variant_prompt": variant,
            "variant_prompt_sha256": content_text_sha256(variant),
        }
        task_input = {
            "pair_id": item_id,
            "baseline_prompt_sha256": payload_content["baseline_prompt_sha256"],
            "variant_prompt_sha256": payload_content["variant_prompt_sha256"],
            "variant_kind": _string(strata.get("variant_kind"), "strata.variant_kind"),
        }
    return payload_content, task_input


def build_item_records(
    authoring: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Convert one approved fixture authoring record into three bound records."""

    _exact(authoring, AUTHORING_FIELDS, "authoring item")
    if authoring["schema_id"] != AUTHORING_SCHEMA_ID:
        raise ValueError("authoring item schema mismatch")
    if authoring["human_approved"] is not True:
        raise ValueError("fixture authoring item requires explicit human approval")
    item_id = _string(authoring["item_id"], "item_id")
    dimension = _string(authoring["dimension"], "dimension")
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError("fixture builder does not support this dimension")
    family_id = _string(authoring["semantic_family_id"], "semantic_family_id")
    parent_id = _string(authoring["parent_document_id"], "parent_document_id")
    strata = _mapping(authoring["strata"], "strata")
    if strata.get("semantic_family") != family_id:
        raise ValueError("strata must bind the exact semantic family")
    content = _mapping(authoring["content"], "content")
    scoring = dict(_mapping(authoring["scoring"], "scoring"))
    payload_content, task_input = _build_content_and_input(
        dimension, content, item_id, strata
    )
    task = {
        "schema_id": TASK_SCHEMA_ID,
        "item_id": item_id,
        "dimension": dimension,
        "split": "development",
        "strata": dict(strata),
        "input": task_input,
        "scoring": scoring,
    }
    validate_task(task)
    payload = {
        "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
        "task": task,
        "content": payload_content,
    }
    validate_payload_record(payload, dimension)

    source = _mapping(authoring["provenance"], "provenance")
    _exact(source, PROVENANCE_INPUT_FIELDS, "provenance")
    provenance = {
        "schema_id": PROVENANCE_RECORD_SCHEMA_ID,
        "item_id": item_id,
        **dict(source),
        "parent_document_id": parent_id,
    }
    validate_provenance_record(provenance)

    prompts = _prompt_values(dimension, payload_content)
    answers = [value for value in _answer_values(dimension, content, scoring) if value]
    semantic_material = {
        "semantic_family_id": family_id,
        "parent_document_id": parent_id,
        "prompts": [normalized_text_sha256(value) for value in prompts],
        "answers": [normalized_text_sha256(value) for value in answers],
    }
    contamination = {
        "schema_id": CONTAMINATION_RECORD_SCHEMA_ID,
        "item_id": item_id,
        "parent_document_id": parent_id,
        "prompt_exact_commitments": _exact_commitments(prompts),
        "answer_exact_commitments": _exact_commitments(answers),
        "ngram_words": 8,
        "ngram_sha256s": _ngram_hashes([*prompts, *answers]),
        "semantic_fingerprint": {
            "method": "declared-family-and-content-sha256-v1",
            "value": hashlib.sha256(canonical_json(semantic_material)).hexdigest(),
        },
    }
    validate_contamination_record(contamination)
    return payload, provenance, contamination


def _jsonl(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json(record) + b"\n" for record in records)


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _binding(path: Path, relative: str, record_count: int) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "record_count": record_count,
        "byte_count": path.stat().st_size,
    }


def build_fixture_inventory(
    *,
    repository_root: Path,
    output_directory: str,
    inventory_id: str,
    suite_id: str,
    repository_commit: str,
    dimension: str,
    scorer: Mapping[str, object],
    authoring_items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build one immutable fixture bundle; production construction is absent."""

    root = repository_root.resolve()
    relative_output = _safe_relative(output_directory, "output_directory")
    unresolved_destination = root / relative_output
    _reject_link_components(unresolved_destination, root, "output directory")
    destination = unresolved_destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError("output directory escapes repository")
    if destination.exists():
        raise FileExistsError("inventory output already exists")
    if not destination.parent.is_dir():
        raise ValueError("inventory output parent must already exist")
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError("stale inventory staging directory exists")
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError("unsupported fixture inventory dimension")
    _string(inventory_id, "inventory_id")
    _string(suite_id, "suite_id")
    _commit(repository_commit, "repository_commit")
    scorer_value = _mapping(scorer, "scorer")
    _exact(scorer_value, {"path", "sha256"}, "scorer")
    scorer_path = _safe_relative(scorer_value["path"], "scorer.path")
    scorer_sha = _sha(scorer_value["sha256"], "scorer.sha256")
    unresolved_scorer = root / scorer_path
    _reject_link_components(unresolved_scorer, root, "scorer path")
    absolute_scorer = unresolved_scorer.resolve()
    if not absolute_scorer.is_relative_to(root) or not absolute_scorer.is_file():
        raise ValueError("scorer path is missing or unsafe")
    if sha256_file(absolute_scorer) != scorer_sha:
        raise ValueError("scorer byte identity mismatch")
    if not authoring_items:
        raise ValueError("fixture inventory requires authoring items")

    records = [build_item_records(item) for item in authoring_items]
    payloads = [record[0] for record in records]
    provenance = [record[1] for record in records]
    contamination = [record[2] for record in records]
    if any(item["task"]["dimension"] != dimension for item in payloads):
        raise ValueError("authoring dimension does not match inventory")
    validate_inventory([item["task"] for item in payloads])
    item_ids = [str(item["task"]["item_id"]) for item in payloads]
    if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("authoring items must use unique sorted IDs")
    family_ids = [str(item["task"]["strata"]["semantic_family"]) for item in payloads]
    parent_ids = [str(item["parent_document_id"]) for item in provenance]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("fixture items must use isolated semantic families")
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("fixture items must use isolated parent documents")

    staging.mkdir()
    try:
        names = {
            "payload": "payload.jsonl",
            "provenance_index": "provenance.jsonl",
            "contamination_index": "contamination.jsonl",
        }
        _write_new(staging / names["payload"], _jsonl(payloads))
        _write_new(staging / names["provenance_index"], _jsonl(provenance))
        _write_new(staging / names["contamination_index"], _jsonl(contamination))
        count = len(payloads)
        manifest: dict[str, object] = {
            "schema_id": INVENTORY_SCHEMA_ID,
            "inventory_id": inventory_id,
            "suite_id": suite_id,
            "repository_commit": repository_commit,
            "dimension": dimension,
            "split": "development",
            "interface": INTERFACES[dimension],
            "tokenizer_sha256": TOKENIZER_SHA256,
            "payload": _binding(
                staging / names["payload"],
                f"{relative_output}/{names['payload']}",
                count,
            ),
            "provenance_index": _binding(
                staging / names["provenance_index"],
                f"{relative_output}/{names['provenance_index']}",
                count,
            ),
            "contamination_index": _binding(
                staging / names["contamination_index"],
                f"{relative_output}/{names['contamination_index']}",
                count,
            ),
            "scorer": {"path": scorer_path, "sha256": scorer_sha},
            "generation_profile_ids": (
                []
                if RESULT_MODES[dimension] in ({"direct_likelihood"}, {"manual"})
                else ["greedy-v1", "sampled-v1"]
            ),
            "item_commitments": [
                {
                    "item_id": item_id,
                    "task_sha256": task_identity(payload["task"]),
                    "provenance_sha256": provenance_identity(provenance_record),
                }
                for item_id, payload, provenance_record in zip(
                    item_ids, payloads, provenance, strict=True
                )
            ],
            "access": {
                "state": "available",
                "payload_format": "jsonl",
                "encryption_algorithm": "none",
                "recipient_fingerprint": None,
                "opening_authorized": False,
            },
            "fixture_only": True,
            "production_suite_frozen": False,
            "evaluation_run_authorized": False,
            "training_authorized": False,
        }
        manifest["inventory_sha256"] = inventory_identity(manifest)
        validate_inventory_manifest(manifest)
        _write_new(staging / "manifest.json", json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8") + b"\n")
        os.replace(staging, destination)
        validate_inventory_manifest_files(manifest, root)
        return manifest
    except Exception:
        # The staging path contains fixture-only bytes and is safe to clean if
        # promotion never occurred. A promoted destination is preserved visibly.
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
        raise


def build_fixture_report(
    manifests: Sequence[Mapping[str, object]],
    *,
    construction_plan: Mapping[str, object],
) -> dict[str, object]:
    """Summarize fixture evidence bound to one validated construction plan."""

    validate_inventory_construction_plan(construction_plan)
    plan_suite_id = str(construction_plan["suite_id"])
    plan_commit = str(construction_plan["repository_commit"])
    plan_sha256 = plan_identity(construction_plan)
    dimensions = [str(manifest["dimension"]) for manifest in manifests]
    if set(dimensions) != SUPPORTED_DIMENSIONS or len(dimensions) != len(
        SUPPORTED_DIMENSIONS
    ):
        raise ValueError("fixture report requires exactly five prompt dimensions")
    for manifest in manifests:
        validate_inventory_manifest(manifest)
        if manifest["fixture_only"] is not True:
            raise ValueError("fixture report cannot include production inventory")
        if manifest["suite_id"] != plan_suite_id:
            raise ValueError("fixture report manifest suite does not match plan")
        if manifest["repository_commit"] != plan_commit:
            raise ValueError("fixture report manifest commit does not match plan")
    report: dict[str, object] = {
        "schema_id": BUILDER_REPORT_SCHEMA_ID,
        "dimensions": sorted(dimensions),
        "inventory_sha256s": {
            str(manifest["dimension"]): str(manifest["inventory_sha256"])
            for manifest in manifests
        },
        "record_counts": {
            str(manifest["dimension"]): int(manifest["payload"]["record_count"])
            for manifest in manifests
        },
        "construction_plan_sha256": plan_sha256,
        "construction_plan_repository_commit": plan_commit,
        "fixture_only": True,
        "production_inventory_created": False,
        "held_out_content_created": False,
        "encryption_key_created": False,
        "model_invoked": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report
