"""Fail-closed schemas for the VASU-140M base-model evaluation v2 suite."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path


SUITE_SCHEMA_ID = "vasu_140m_base_model_evaluation_suite_v2"
RESULT_SCHEMA_ID = "vasu_140m_base_model_evaluation_result_v2"
FAMILY_ID = "vasu_140m_v1"
MODEL_CONFIG_SHA256 = (
    "29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059"
)
TOKENIZER_SHA256 = (
    "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
)
DIMENSIONS = frozenset(
    {
        "likelihood",
        "factuality",
        "arithmetic",
        "repetition",
        "robustness",
        "manual_review",
    }
)
SPLITS = frozenset({"development", "held_out"})
INTERFACES = {
    "likelihood": "direct_likelihood",
    "factuality": "direct_likelihood",
    "arithmetic": "raw_continuation",
    "repetition": "raw_continuation",
    "robustness": "paired_continuation",
    "manual_review": "raw_continuation",
}
RESULT_MODES = {
    "likelihood": frozenset({"direct_likelihood"}),
    "factuality": frozenset({"direct_likelihood"}),
    "arithmetic": frozenset({"greedy", "sampled"}),
    "repetition": frozenset({"greedy", "sampled"}),
    "robustness": frozenset({"greedy", "sampled"}),
    "manual_review": frozenset({"manual"}),
}
SHA256_HEX_LENGTH = 64


def canonical_json(value: object) -> bytes:
    """Encode identity material with one deterministic JSON representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: Mapping[str, object], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def suite_identity(manifest: Mapping[str, object]) -> str:
    return _identity(manifest, "suite_sha256")


def result_identity(result: Mapping[str, object]) -> str:
    return _identity(result, "result_sha256")


def inventory_index_identity(inventories: object) -> str:
    return hashlib.sha256(canonical_json(inventories)).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
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


def _git_commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return text


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {comparator} integer")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _timestamp(value: object, label: str) -> str:
    text = _string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return text


def _validate_bound_path(value: object, label: str) -> tuple[str, str]:
    binding = _mapping(value, label)
    _exact_keys(binding, {"path", "sha256"}, label)
    return _safe_path(binding["path"], f"{label}.path"), _sha256(
        binding["sha256"], f"{label}.sha256"
    )


def _validate_generation_profile(value: object, index: int) -> str:
    label = f"generation profile {index}"
    profile = _mapping(value, label)
    _exact_keys(
        profile,
        {
            "profile_id",
            "decoding",
            "max_new_tokens",
            "seeds",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "eos_token_id",
            "pad_token_id",
            "batch_size",
            "precision",
            "kv_cache",
        },
        label,
    )
    profile_id = _string(profile["profile_id"], f"{label}.profile_id")
    if profile["decoding"] not in {"greedy", "sampled"}:
        raise ValueError(f"{label}.decoding is unsupported")
    _positive_int(profile["max_new_tokens"], f"{label}.max_new_tokens")
    seeds = profile["seeds"]
    if not isinstance(seeds, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in seeds
    ):
        raise ValueError(f"{label}.seeds must contain non-negative integers")
    if profile["decoding"] == "greedy" and seeds:
        raise ValueError("greedy generation must not declare sampling seeds")
    if profile["decoding"] == "sampled" and not seeds:
        raise ValueError("sampled generation requires at least one seed")
    temperature = _finite_number(profile["temperature"], f"{label}.temperature")
    top_p = _finite_number(profile["top_p"], f"{label}.top_p")
    penalty = _finite_number(
        profile["repetition_penalty"], f"{label}.repetition_penalty"
    )
    _positive_int(profile["top_k"], f"{label}.top_k", allow_zero=True)
    _positive_int(profile["eos_token_id"], f"{label}.eos_token_id", allow_zero=True)
    _positive_int(profile["pad_token_id"], f"{label}.pad_token_id", allow_zero=True)
    _positive_int(profile["batch_size"], f"{label}.batch_size")
    if temperature < 0 or not 0 < top_p <= 1 or penalty <= 0:
        raise ValueError(f"{label} contains invalid decoding thresholds")
    if profile["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"{label}.precision is unsupported")
    if not isinstance(profile["kv_cache"], bool):
        raise ValueError(f"{label}.kv_cache must be boolean")
    return profile_id


def validate_suite_manifest(manifest: Mapping[str, object]) -> None:
    """Validate a frozen suite manifest without opening any inventory files."""

    _exact_keys(
        manifest,
        {
            "schema_id",
            "suite_id",
            "repository_commit",
            "family_id",
            "model_config_sha256",
            "tokenizer",
            "inventories",
            "scorers",
            "generation_profiles",
            "bootstrap",
            "contamination",
            "held_out_policy",
            "training_authorized",
            "suite_sha256",
        },
        "suite manifest",
    )
    if manifest["schema_id"] != SUITE_SCHEMA_ID:
        raise ValueError("suite schema identity mismatch")
    _string(manifest["suite_id"], "suite_id")
    _git_commit(manifest["repository_commit"], "repository_commit")
    if manifest["family_id"] != FAMILY_ID:
        raise ValueError("suite family identity mismatch")
    if _sha256(manifest["model_config_sha256"], "model_config_sha256") != (
        MODEL_CONFIG_SHA256
    ):
        raise ValueError("suite model configuration identity mismatch")
    _, tokenizer_sha = _validate_bound_path(manifest["tokenizer"], "tokenizer")
    if tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError("suite tokenizer identity mismatch")
    if manifest["training_authorized"] is not False:
        raise ValueError("training_authorized must be false")

    scorer_ids: set[str] = set()
    scorer_dimensions: set[str] = set()
    for index, item in enumerate(_list(manifest["scorers"], "scorers")):
        label = f"scorer {index}"
        scorer = _mapping(item, label)
        _exact_keys(
            scorer,
            {"scorer_id", "dimension", "metric_kind", "path", "sha256"},
            label,
        )
        scorer_id = _string(scorer["scorer_id"], f"{label}.scorer_id")
        dimension = _string(scorer["dimension"], f"{label}.dimension")
        if scorer_id in scorer_ids or dimension in scorer_dimensions:
            raise ValueError("scorers must have unique IDs and dimensions")
        if dimension not in DIMENSIONS:
            raise ValueError(f"{label}.dimension is unsupported")
        expected_kind = "manual" if dimension == "manual_review" else "objective"
        if scorer["metric_kind"] != expected_kind:
            raise ValueError(f"{label}.metric_kind does not match its dimension")
        _safe_path(scorer["path"], f"{label}.path")
        _sha256(scorer["sha256"], f"{label}.sha256")
        scorer_ids.add(scorer_id)
        scorer_dimensions.add(dimension)
    if scorer_dimensions != DIMENSIONS:
        raise ValueError("scorers must cover every evaluation dimension exactly once")

    profile_ids: set[str] = set()
    profile_decoding: dict[str, str] = {}
    for index, item in enumerate(
        _list(manifest["generation_profiles"], "generation_profiles")
    ):
        profile_id = _validate_generation_profile(item, index)
        if profile_id in profile_ids:
            raise ValueError("generation profile IDs must be unique")
        profile_ids.add(profile_id)
        profile_decoding[profile_id] = item["decoding"]
    if set(profile_decoding.values()) != {"greedy", "sampled"}:
        raise ValueError("generation profiles must include greedy and sampled modes")

    inventories = _list(manifest["inventories"], "inventories")
    inventory_ids: set[str] = set()
    coverage: set[tuple[str, str]] = set()
    for index, item in enumerate(inventories):
        label = f"inventory {index}"
        inventory = _mapping(item, label)
        _exact_keys(
            inventory,
            {
                "inventory_id",
                "dimension",
                "split",
                "path",
                "sha256",
                "provenance_sha256",
                "item_count",
                "interface",
                "scorer_id",
                "generation_profile_ids",
                "access_state",
            },
            label,
        )
        inventory_id = _string(
            inventory["inventory_id"], f"{label}.inventory_id"
        )
        dimension = _string(inventory["dimension"], f"{label}.dimension")
        split = _string(inventory["split"], f"{label}.split")
        if inventory_id in inventory_ids:
            raise ValueError("inventory IDs must be unique")
        if dimension not in DIMENSIONS or split not in SPLITS:
            raise ValueError(f"{label} has unsupported dimension/split")
        if (dimension, split) in coverage:
            raise ValueError("each dimension/split inventory must be unique")
        if inventory["interface"] != INTERFACES[dimension]:
            raise ValueError(f"{label}.interface does not match its dimension")
        if inventory["scorer_id"] not in scorer_ids:
            raise ValueError(f"{label}.scorer_id is unknown")
        scorer = next(
            candidate
            for candidate in manifest["scorers"]
            if candidate["scorer_id"] == inventory["scorer_id"]
        )
        if scorer["dimension"] != dimension:
            raise ValueError(f"{label}.scorer_id belongs to another dimension")
        referenced_profiles = inventory["generation_profile_ids"]
        if not isinstance(referenced_profiles, list) or any(
            not isinstance(profile_id, str) for profile_id in referenced_profiles
        ):
            raise ValueError(f"{label}.generation_profile_ids must be a string list")
        if len(referenced_profiles) != len(set(referenced_profiles)):
            raise ValueError(f"{label}.generation_profile_ids contains duplicates")
        if INTERFACES[dimension] == "direct_likelihood":
            if referenced_profiles:
                raise ValueError("direct-likelihood inventories cannot generate text")
        else:
            if not referenced_profiles or set(referenced_profiles) - profile_ids:
                raise ValueError(f"{label}.generation_profile_ids are missing or unknown")
            decoding_modes = {
                profile_decoding[profile_id] for profile_id in referenced_profiles
            }
            if decoding_modes != {"greedy", "sampled"}:
                raise ValueError(
                    f"{label} must bind both greedy and sampled generation"
                )
        expected_access = "sealed" if split == "held_out" else "available"
        if inventory["access_state"] != expected_access:
            raise ValueError(f"{label}.access_state must be {expected_access}")
        _safe_path(inventory["path"], f"{label}.path")
        _sha256(inventory["sha256"], f"{label}.sha256")
        _sha256(inventory["provenance_sha256"], f"{label}.provenance_sha256")
        _positive_int(inventory["item_count"], f"{label}.item_count")
        inventory_ids.add(inventory_id)
        coverage.add((dimension, split))
    expected_coverage = {(dimension, split) for dimension in DIMENSIONS for split in SPLITS}
    if coverage != expected_coverage:
        raise ValueError("inventories must cover every dimension in both splits")

    bootstrap = _mapping(manifest["bootstrap"], "bootstrap")
    _exact_keys(bootstrap, {"confidence_level", "resamples", "seed"}, "bootstrap")
    confidence = _finite_number(bootstrap["confidence_level"], "bootstrap.confidence_level")
    if confidence != 0.95:
        raise ValueError("bootstrap.confidence_level must be 0.95")
    if _positive_int(bootstrap["resamples"], "bootstrap.resamples") < 1000:
        raise ValueError("bootstrap.resamples must be at least 1000")
    _positive_int(bootstrap["seed"], "bootstrap.seed", allow_zero=True)

    contamination = _mapping(manifest["contamination"], "contamination")
    _exact_keys(
        contamination,
        {
            "inventory_index_sha256",
            "exact_method",
            "ngram_words",
            "semantic_method",
            "scan_before_training_split",
        },
        "contamination",
    )
    if _sha256(
        contamination["inventory_index_sha256"],
        "contamination.inventory_index_sha256",
    ) != inventory_index_identity(inventories):
        raise ValueError("contamination inventory commitment mismatch")
    _string(contamination["exact_method"], "contamination.exact_method")
    if _positive_int(contamination["ngram_words"], "contamination.ngram_words") < 8:
        raise ValueError("contamination.ngram_words must be at least 8")
    _string(contamination["semantic_method"], "contamination.semantic_method")
    if contamination["scan_before_training_split"] is not True:
        raise ValueError("contamination scan must run before training split assignment")

    held_out = _mapping(manifest["held_out_policy"], "held_out_policy")
    _exact_keys(
        held_out,
        {"decision_point", "opened", "opening_authorized"},
        "held_out_policy",
    )
    _string(held_out["decision_point"], "held_out_policy.decision_point")
    if held_out["opened"] is not False or held_out["opening_authorized"] is not False:
        raise ValueError("held-out inventories must remain sealed and unauthorized")

    if _sha256(manifest["suite_sha256"], "suite_sha256") != suite_identity(manifest):
        raise ValueError("suite manifest identity mismatch")


def validate_suite_manifest_files(
    manifest: Mapping[str, object],
    repository_root: Path,
    *,
    open_held_out: bool = False,
) -> None:
    """Verify repository-bound files while refusing held-out access by default."""

    validate_suite_manifest(manifest)
    if open_held_out:
        raise PermissionError("held-out access requires a future reviewed decision gate")
    root = repository_root.resolve()
    tokenizer_path, tokenizer_sha = _validate_bound_path(manifest["tokenizer"], "tokenizer")
    _verify_file(root, tokenizer_path, tokenizer_sha, "tokenizer")
    for index, raw in enumerate(manifest["scorers"]):
        scorer = _mapping(raw, f"scorer {index}")
        _verify_file(
            root,
            _safe_path(scorer["path"], f"scorer {index}.path"),
            _sha256(scorer["sha256"], f"scorer {index}.sha256"),
            f"scorer {index}",
        )
    for index, raw in enumerate(manifest["inventories"]):
        inventory = _mapping(raw, f"inventory {index}")
        if inventory["split"] == "held_out":
            continue
        _verify_file(
            root,
            _safe_path(inventory["path"], f"inventory {index}.path"),
            _sha256(inventory["sha256"], f"inventory {index}.sha256"),
            f"inventory {index}",
        )


def _verify_file(root: Path, relative: str, expected_sha: str, label: str) -> None:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} path escapes the repository")
    if not path.is_file():
        raise ValueError(f"{label} path does not exist")
    if sha256_file(path) != expected_sha:
        raise ValueError(f"{label} file identity mismatch")


def resume_identity(result: Mapping[str, object]) -> str:
    suite = _mapping(result["suite"], "suite")
    checkpoint = _mapping(result["checkpoint"], "checkpoint")
    tokenizer = _mapping(result["tokenizer"], "tokenizer")
    runtime = _mapping(result["runtime"], "runtime")
    held_out_authorization = _mapping(
        result["held_out_authorization"], "held_out_authorization"
    )
    payload = {
        "suite_sha256": suite["sha256"],
        "checkpoint_sha256": checkpoint["sha256"],
        "tokenizer_sha256": tokenizer["sha256"],
        "repository_commit": runtime["repository_commit"],
        "environment_sha256": runtime["environment_sha256"],
        "device": runtime["device"],
        "precision": runtime["precision"],
        "command": runtime["command"],
        "evaluation_stage": result["evaluation_stage"],
        "held_out_authorization_sha256": held_out_authorization["sha256"],
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def validate_result_manifest(result: Mapping[str, object]) -> None:
    """Validate complete model-result identity without accepting an aggregate score."""

    _exact_keys(
        result,
        {
            "schema_id",
            "result_id",
            "suite",
            "checkpoint",
            "tokenizer",
            "evaluation_stage",
            "held_out_authorization",
            "runtime",
            "output_shards",
            "dimension_reports",
            "resume_identity_sha256",
            "training_authorized",
            "result_sha256",
        },
        "result manifest",
    )
    if result["schema_id"] != RESULT_SCHEMA_ID:
        raise ValueError("result schema identity mismatch")
    _string(result["result_id"], "result_id")
    _, suite_sha = _validate_bound_path(result["suite"], "suite")
    checkpoint = _mapping(result["checkpoint"], "checkpoint")
    _exact_keys(
        checkpoint,
        {"path", "sha256", "family_id", "model_config_sha256"},
        "checkpoint",
    )
    _safe_path(checkpoint["path"], "checkpoint.path")
    _sha256(checkpoint["sha256"], "checkpoint.sha256")
    if checkpoint["family_id"] != FAMILY_ID:
        raise ValueError("result checkpoint family mismatch")
    if _sha256(
        checkpoint["model_config_sha256"], "checkpoint.model_config_sha256"
    ) != MODEL_CONFIG_SHA256:
        raise ValueError("result checkpoint model configuration mismatch")
    _, tokenizer_sha = _validate_bound_path(result["tokenizer"], "tokenizer")
    if tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError("result tokenizer identity mismatch")

    evaluation_stage = _string(result["evaluation_stage"], "evaluation_stage")
    if evaluation_stage not in SPLITS:
        raise ValueError("evaluation_stage is unsupported")
    held_out_authorization = _mapping(
        result["held_out_authorization"], "held_out_authorization"
    )
    _exact_keys(
        held_out_authorization,
        {"authorized", "decision_id", "path", "sha256"},
        "held_out_authorization",
    )
    if evaluation_stage == "development":
        if held_out_authorization != {
            "authorized": False,
            "decision_id": None,
            "path": None,
            "sha256": None,
        }:
            raise ValueError(
                "development results must not declare held-out authorization"
            )
    else:
        if held_out_authorization["authorized"] is not True:
            raise ValueError("held-out results require explicit authorization")
        _string(
            held_out_authorization["decision_id"],
            "held_out_authorization.decision_id",
        )
        _safe_path(
            held_out_authorization["path"], "held_out_authorization.path"
        )
        _sha256(
            held_out_authorization["sha256"], "held_out_authorization.sha256"
        )

    runtime = _mapping(result["runtime"], "runtime")
    _exact_keys(
        runtime,
        {
            "repository_commit",
            "environment_sha256",
            "device",
            "precision",
            "command",
            "started_at",
            "completed_at",
        },
        "runtime",
    )
    _git_commit(runtime["repository_commit"], "runtime.repository_commit")
    _sha256(runtime["environment_sha256"], "runtime.environment_sha256")
    _string(runtime["device"], "runtime.device")
    if runtime["precision"] not in {"fp32", "bf16", "fp16"}:
        raise ValueError("runtime.precision is unsupported")
    command = runtime["command"]
    if not isinstance(command, list) or not command or any(
        not isinstance(part, str) or not part for part in command
    ):
        raise ValueError("runtime.command must be a non-empty string list")
    started = datetime.fromisoformat(_timestamp(runtime["started_at"], "runtime.started_at"))
    completed = datetime.fromisoformat(
        _timestamp(runtime["completed_at"], "runtime.completed_at")
    )
    if completed < started:
        raise ValueError("runtime.completed_at precedes runtime.started_at")

    shard_coverage: set[tuple[str, str]] = set()
    shard_ids: set[str] = set()
    shard_paths: set[str] = set()
    for index, raw in enumerate(_list(result["output_shards"], "output_shards")):
        label = f"output shard {index}"
        shard = _mapping(raw, label)
        _exact_keys(
            shard,
            {
                "shard_id",
                "dimension",
                "split",
                "mode",
                "path",
                "sha256",
                "record_count",
            },
            label,
        )
        shard_id = _string(shard["shard_id"], f"{label}.shard_id")
        dimension = _string(shard["dimension"], f"{label}.dimension")
        if dimension not in DIMENSIONS:
            raise ValueError("output shard dimension is unsupported")
        if shard["split"] != evaluation_stage:
            raise ValueError(f"{label}.split does not match evaluation_stage")
        if shard["mode"] not in RESULT_MODES[dimension]:
            raise ValueError(f"{label}.mode does not match its dimension")
        shard_path = _safe_path(shard["path"], f"{label}.path")
        if shard_id in shard_ids or shard_path in shard_paths:
            raise ValueError("output shard IDs and paths must be unique")
        _sha256(shard["sha256"], f"{label}.sha256")
        _positive_int(shard["record_count"], f"{label}.record_count")
        shard_ids.add(shard_id)
        shard_paths.add(shard_path)
        shard_coverage.add((dimension, shard["mode"]))
    expected_result_coverage = {
        (dimension, mode)
        for dimension, modes in RESULT_MODES.items()
        for mode in modes
    }
    if shard_coverage != expected_result_coverage:
        raise ValueError(
            "output shards must cover every required dimension/mode pair"
        )

    report_coverage: set[tuple[str, str]] = set()
    report_keys: set[tuple[str, str, str, str]] = set()
    for index, raw in enumerate(
        _list(result["dimension_reports"], "dimension_reports")
    ):
        label = f"dimension report {index}"
        report = _mapping(raw, label)
        _exact_keys(
            report,
            {
                "dimension",
                "split",
                "mode",
                "metric_name",
                "numerator",
                "denominator",
                "point_estimate",
                "confidence_interval_95",
                "strata_path",
                "strata_sha256",
            },
            label,
        )
        dimension = _string(report["dimension"], f"{label}.dimension")
        if dimension not in DIMENSIONS:
            raise ValueError("dimension report is unsupported")
        if report["split"] != evaluation_stage:
            raise ValueError(f"{label}.split does not match evaluation_stage")
        mode = _string(report["mode"], f"{label}.mode")
        if mode not in RESULT_MODES[dimension]:
            raise ValueError(f"{label}.mode does not match its dimension")
        metric_name = _string(report["metric_name"], f"{label}.metric_name")
        report_key = (dimension, evaluation_stage, mode, metric_name)
        if report_key in report_keys:
            raise ValueError("dimension/metric report pairs must be unique")
        numerator = _finite_number(report["numerator"], f"{label}.numerator")
        if numerator < 0:
            raise ValueError(f"{label}.numerator must be non-negative")
        _positive_int(report["denominator"], f"{label}.denominator")
        point = _finite_number(report["point_estimate"], f"{label}.point_estimate")
        interval = report["confidence_interval_95"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"{label}.confidence_interval_95 must have two bounds")
        lower = _finite_number(interval[0], f"{label}.confidence_interval_95[0]")
        upper = _finite_number(interval[1], f"{label}.confidence_interval_95[1]")
        if lower > point or point > upper:
            raise ValueError(f"{label} point estimate must lie inside its interval")
        _safe_path(report["strata_path"], f"{label}.strata_path")
        _sha256(report["strata_sha256"], f"{label}.strata_sha256")
        report_keys.add(report_key)
        report_coverage.add((dimension, mode))
    if report_coverage != expected_result_coverage:
        raise ValueError(
            "dimension reports must cover every required dimension/mode pair"
        )
    if shard_coverage != report_coverage:
        raise ValueError("output shard/report dimension-mode coverage differs")
    if _sha256(result["resume_identity_sha256"], "resume_identity_sha256") != (
        resume_identity(result)
    ):
        raise ValueError("result resume identity mismatch")
    if result["training_authorized"] is not False:
        raise ValueError("training_authorized must be false")
    if _sha256(result["result_sha256"], "result_sha256") != result_identity(result):
        raise ValueError("result manifest identity mismatch")
    _sha256(suite_sha, "suite.sha256")


def validate_result_against_suite(
    result: Mapping[str, object], suite: Mapping[str, object]
) -> None:
    """Verify that one result is bound to the exact validated suite identity."""

    validate_suite_manifest(suite)
    validate_result_manifest(result)
    result_suite = _mapping(result["suite"], "suite")
    if result_suite["sha256"] != suite["suite_sha256"]:
        raise ValueError("result suite identity does not match suite manifest")
    result_tokenizer = _mapping(result["tokenizer"], "tokenizer")
    suite_tokenizer = _mapping(suite["tokenizer"], "tokenizer")
    if result_tokenizer["sha256"] != suite_tokenizer["sha256"]:
        raise ValueError("result tokenizer does not match suite manifest")


def build_synthetic_suite_manifest(
    *,
    repository_commit: str,
    tokenizer_path: str,
    tokenizer_sha256: str,
    scorer_path: str,
    scorer_sha256: str,
    development_inventory_path: str,
    development_inventory_sha256: str,
) -> dict[str, object]:
    """Build prompt-free fixture material for schema qualification only."""

    scorers = [
        {
            "scorer_id": f"{dimension}-scorer-v1",
            "dimension": dimension,
            "metric_kind": (
                "manual" if dimension == "manual_review" else "objective"
            ),
            "path": scorer_path,
            "sha256": scorer_sha256,
        }
        for dimension in sorted(DIMENSIONS)
    ]
    inventories: list[dict[str, object]] = []
    for dimension in sorted(DIMENSIONS):
        for split in ("development", "held_out"):
            inventories.append(
                {
                    "inventory_id": f"{dimension}-{split}-synthetic-v1",
                    "dimension": dimension,
                    "split": split,
                    "path": (
                        development_inventory_path
                        if split == "development"
                        else f"evaluation/held_out/sealed/{dimension}.jsonl"
                    ),
                    "sha256": (
                        development_inventory_sha256
                        if split == "development"
                        else "d" * 64
                    ),
                    "provenance_sha256": "e" * 64,
                    "item_count": 2,
                    "interface": INTERFACES[dimension],
                    "scorer_id": f"{dimension}-scorer-v1",
                    "generation_profile_ids": (
                        []
                        if INTERFACES[dimension] == "direct_likelihood"
                        else ["greedy-v1", "sampled-v1"]
                    ),
                    "access_state": (
                        "sealed" if split == "held_out" else "available"
                    ),
                }
            )
    manifest: dict[str, object] = {
        "schema_id": SUITE_SCHEMA_ID,
        "suite_id": "vasu-140m-base-eval-v2-synthetic",
        "repository_commit": repository_commit,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer": {"path": tokenizer_path, "sha256": tokenizer_sha256},
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
        "bootstrap": {
            "confidence_level": 0.95,
            "resamples": 10_000,
            "seed": 140,
        },
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
    manifest["suite_sha256"] = suite_identity(manifest)
    return manifest


def build_synthetic_result_manifest(
    *, repository_commit: str, suite_sha256: str
) -> dict[str, object]:
    """Build checkpoint-free synthetic result metadata for validation tests."""

    result: dict[str, object] = {
        "schema_id": RESULT_SCHEMA_ID,
        "result_id": "vasu-140m-base-eval-v2-synthetic-result",
        "suite": {
            "path": "evaluation/suites/synthetic.json",
            "sha256": suite_sha256,
        },
        "checkpoint": {
            "path": "checkpoints/vasu_140m/synthetic.pt",
            "sha256": "2" * 64,
            "family_id": FAMILY_ID,
            "model_config_sha256": MODEL_CONFIG_SHA256,
        },
        "tokenizer": {
            "path": "assets/tokenizer.json",
            "sha256": TOKENIZER_SHA256,
        },
        "evaluation_stage": "development",
        "held_out_authorization": {
            "authorized": False,
            "decision_id": None,
            "path": None,
            "sha256": None,
        },
        "runtime": {
            "repository_commit": repository_commit,
            "environment_sha256": "3" * 64,
            "device": "cuda:0",
            "precision": "bf16",
            "command": ["python", "scripts/evaluate_vasu_140m_base_v2.py"],
            "started_at": "2026-08-01T12:00:00+05:30",
            "completed_at": "2026-08-01T12:30:00+05:30",
        },
        "output_shards": [
            {
                "shard_id": f"{dimension}-development-{mode}",
                "dimension": dimension,
                "split": "development",
                "mode": mode,
                "path": (
                    f"evaluation/results/synthetic/{dimension}_{mode}.jsonl"
                ),
                "sha256": "4" * 64,
                "record_count": 2,
            }
            for dimension in sorted(DIMENSIONS)
            for mode in sorted(RESULT_MODES[dimension])
        ],
        "dimension_reports": [
            {
                "dimension": dimension,
                "split": "development",
                "mode": mode,
                "metric_name": f"{dimension}_{mode}_primary",
                "numerator": 1,
                "denominator": 2,
                "point_estimate": 0.5,
                "confidence_interval_95": [0.0, 1.0],
                "strata_path": (
                    f"evaluation/results/synthetic/{dimension}_{mode}_strata.json"
                ),
                "strata_sha256": "5" * 64,
            }
            for dimension in sorted(DIMENSIONS)
            for mode in sorted(RESULT_MODES[dimension])
        ],
        "resume_identity_sha256": "0" * 64,
        "training_authorized": False,
        "result_sha256": "0" * 64,
    }
    result["resume_identity_sha256"] = resume_identity(result)
    result["result_sha256"] = result_identity(result)
    return result
