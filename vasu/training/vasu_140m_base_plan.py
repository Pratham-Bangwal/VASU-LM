"""Strict review-only schema for a future VASU-140M base-pretraining plan."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path


SCHEMA_ID = "vasu_140m_base_pretraining_plan_v2"
FAMILY_ID = "vasu_140m_v1"
FAMILY_SHA256 = "72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b"
MODEL_CONFIG_SHA256 = "29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059"
TOKENIZER_SHA256 = "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
RECORD_WIDTH = 513
SEQUENCE_LENGTH = 512
REQUIRED_GATES = frozenset(
    {"cuda_amp", "base_data_release", "base_evaluation_v2", "real_data_exact_resume"}
)
REQUIRED_DIMENSIONS = frozenset(
    {
        "likelihood",
        "factuality",
        "arithmetic",
        "repetition",
        "robustness",
        "manual_review",
    }
)
_OPERATORS = frozenset({">", ">=", "<", "<=", "=="})


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
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


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return text


def _positive_int(value: object, label: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        adjective = "non-negative" if zero else "positive"
        raise ValueError(f"{label} must be a {adjective} integer")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{label} must be {'positive and ' if positive else ''}finite")
    return number


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return text


def _binding(value: object, label: str) -> tuple[str, str]:
    item = _mapping(value, label)
    _exact(item, {"path", "sha256"}, label)
    return _safe_path(item["path"], f"{label}.path"), _sha(
        item["sha256"], f"{label}.sha256"
    )


def _validate_gate_evidence(
    value: object,
) -> dict[str, tuple[tuple[str, str], tuple[str, str]]]:
    gates = _mapping(value, "gate_evidence")
    if set(gates) != REQUIRED_GATES:
        raise ValueError("gate_evidence must contain exactly the four readiness gates")
    identities: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {}
    bound_paths: set[str] = set()
    for name, raw in gates.items():
        gate = _mapping(raw, f"gate {name}")
        _exact(gate, {"artifact", "decision", "status"}, f"gate {name}")
        artifact = _binding(gate["artifact"], f"gate {name}.artifact")
        decision = _binding(gate["decision"], f"gate {name}.decision")
        artifact_path = artifact[0].casefold()
        decision_path = decision[0].casefold()
        if artifact_path == decision_path:
            raise ValueError(f"gate {name} artifact and decision must be distinct")
        if artifact_path in bound_paths or decision_path in bound_paths:
            raise ValueError("gate artifact and decision paths must be unique")
        if gate["status"] != "accepted":
            raise ValueError(f"gate {name} must be independently accepted")
        bound_paths.update((artifact_path, decision_path))
        identities[name] = (artifact, decision)
    return identities


def _validate_sources(value: object) -> tuple[set[str], int, int]:
    if not isinstance(value, list) or not value:
        raise ValueError("data.sources must be a non-empty list")
    source_ids: set[str] = set()
    records = 0
    supervised = 0
    for index, raw in enumerate(value):
        label = f"data source {index}"
        source = _mapping(raw, label)
        _exact(
            source,
            {
                "source_id",
                "manifest_sha256",
                "token_sha256",
                "mask_sha256",
                "lineage_sha256",
                "train_record_count",
                "supervised_target_count",
            },
            label,
        )
        source_id = _string(source["source_id"], f"{label}.source_id")
        if source_id in source_ids:
            raise ValueError("data source IDs must be unique")
        for field in (
            "manifest_sha256",
            "token_sha256",
            "mask_sha256",
            "lineage_sha256",
        ):
            _sha(source[field], f"{label}.{field}")
        records += _positive_int(
            source["train_record_count"], f"{label}.train_record_count"
        )
        supervised += _positive_int(
            source["supervised_target_count"], f"{label}.supervised_target_count"
        )
        source_ids.add(source_id)
    return source_ids, records, supervised


def _validate_rule_list(value: object, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        rule_label = f"{label} rule {index}"
        rule = _mapping(raw, rule_label)
        _exact(rule, {"metric", "scope", "operator", "threshold"}, rule_label)
        metric = _string(rule["metric"], f"{rule_label}.metric")
        scope = _string(rule["scope"], f"{rule_label}.scope")
        if (metric, scope) in seen:
            raise ValueError(f"{label} metric/scope pairs must be unique")
        if rule["operator"] not in _OPERATORS:
            raise ValueError(f"{rule_label}.operator is unsupported")
        _finite(rule["threshold"], f"{rule_label}.threshold")
        seen.add((metric, scope))


def validate_base_pretraining_plan(plan: Mapping[str, object]) -> None:
    """Validate a non-authorizing future plan without making it launchable."""

    _exact(
        plan,
        {
            "schema_id",
            "experiment_id",
            "repository_commit",
            "family_id",
            "family_sha256",
            "model_config_sha256",
            "tokenizer",
            "gate_evidence",
            "hypothesis",
            "initialization",
            "data",
            "training",
            "evaluation",
            "runtime_envelope",
            "safety",
            "outputs",
            "review",
            "training_authorized",
            "plan_sha256",
        },
        "base-pretraining plan",
    )
    if plan["schema_id"] != SCHEMA_ID or plan["family_id"] != FAMILY_ID:
        raise ValueError("plan schema/family identity mismatch")
    _string(plan["experiment_id"], "experiment_id")
    _commit(plan["repository_commit"], "repository_commit")
    if _sha(plan["family_sha256"], "family_sha256") != FAMILY_SHA256:
        raise ValueError("plan family fingerprint mismatch")
    if _sha(plan["model_config_sha256"], "model_config_sha256") != MODEL_CONFIG_SHA256:
        raise ValueError("plan model configuration mismatch")
    _, tokenizer_sha = _binding(plan["tokenizer"], "tokenizer")
    if tokenizer_sha != TOKENIZER_SHA256:
        raise ValueError("plan tokenizer identity mismatch")
    gate_identities = _validate_gate_evidence(plan["gate_evidence"])

    hypothesis = _mapping(plan["hypothesis"], "hypothesis")
    _exact(
        hypothesis,
        {"question", "prediction", "falsifier", "comparison"},
        "hypothesis",
    )
    for field in ("question", "prediction", "falsifier"):
        _string(hypothesis[field], f"hypothesis.{field}")
    comparison = _mapping(hypothesis["comparison"], "hypothesis.comparison")
    _exact(
        comparison,
        {"kind", "reference", "scientific_role", "matched_parent"},
        "hypothesis.comparison",
    )
    if comparison["kind"] != "pre_update_random_initialization":
        raise ValueError("first-lineage comparison must use the pre-update model")
    _string(comparison["reference"], "hypothesis.comparison.reference")
    _string(comparison["scientific_role"], "hypothesis.comparison.scientific_role")
    if comparison["matched_parent"] is not False:
        raise ValueError("first VASU-140M run has no matched trained parent")

    initialization = _mapping(plan["initialization"], "initialization")
    _exact(
        initialization,
        {"kind", "parent_checkpoint", "seed", "implementation"},
        "initialization",
    )
    if (
        initialization["kind"] != "fresh_random"
        or initialization["parent_checkpoint"] is not None
    ):
        raise ValueError(
            "first VASU-140M base training must use fresh random initialization"
        )
    _positive_int(initialization["seed"], "initialization.seed", zero=True)
    _binding(initialization["implementation"], "initialization.implementation")

    data = _mapping(plan["data"], "data")
    _exact(
        data,
        {
            "release",
            "schedule",
            "sources",
            "no_replacement",
            "record_width",
            "sequence_length",
        },
        "data",
    )
    release_binding = _binding(data["release"], "data.release")
    if release_binding != gate_identities["base_data_release"][0]:
        raise ValueError("selected data release does not match accepted gate evidence")
    schedule = _mapping(data["schedule"], "data.schedule")
    _exact(schedule, {"path", "sha256", "record_count"}, "data.schedule")
    _safe_path(schedule["path"], "data.schedule.path")
    _sha(schedule["sha256"], "data.schedule.sha256")
    if schedule["path"].replace("\\", "/") == release_binding[0]:
        raise ValueError("data schedule must be distinct from the release manifest")
    schedule_records = _positive_int(
        schedule["record_count"], "data.schedule.record_count"
    )
    _, source_records, source_supervised = _validate_sources(data["sources"])
    if data["no_replacement"] is not True:
        raise ValueError("base-pretraining schedule must prohibit replacement")
    if (
        data["record_width"] != RECORD_WIDTH
        or data["sequence_length"] != SEQUENCE_LENGTH
    ):
        raise ValueError("base-pretraining record/context identity mismatch")
    if schedule_records != source_records:
        raise ValueError("schedule and source record counts differ")

    training = _mapping(plan["training"], "training")
    _exact(
        training,
        {
            "batch_size",
            "gradient_accumulation_steps",
            "microbatch_count",
            "optimizer_updates",
            "processed_records",
            "processed_input_positions",
            "supervised_target_budget",
            "precision",
            "seed",
            "gradient_clip",
            "optimizer",
            "scheduler",
            "validation_interval_updates",
            "checkpoint_interval_updates",
            "exact_resume",
            "final_validation",
            "final_checkpoint",
        },
        "training",
    )
    batch = _positive_int(training["batch_size"], "training.batch_size")
    accumulation = _positive_int(
        training["gradient_accumulation_steps"], "training.gradient_accumulation_steps"
    )
    processed_records = _positive_int(
        training["processed_records"], "training.processed_records"
    )
    microbatches = _positive_int(
        training["microbatch_count"], "training.microbatch_count"
    )
    updates = _positive_int(training["optimizer_updates"], "training.optimizer_updates")
    if processed_records != schedule_records:
        raise ValueError("training and schedule record counts differ")
    if processed_records % batch or microbatches != processed_records // batch:
        raise ValueError("microbatch accounting mismatch")
    if microbatches % accumulation or updates != microbatches // accumulation:
        raise ValueError("optimizer-update accounting mismatch")
    if training["processed_input_positions"] != processed_records * SEQUENCE_LENGTH:
        raise ValueError("processed input-position budget mismatch")
    if training["supervised_target_budget"] != source_supervised:
        raise ValueError("supervised-target budget mismatch")
    if training["precision"] not in {"bf16", "fp16"}:
        raise ValueError("training precision must be accepted CUDA AMP precision")
    _positive_int(training["seed"], "training.seed", zero=True)
    _finite(training["gradient_clip"], "training.gradient_clip", positive=True)
    _positive_int(
        training["validation_interval_updates"], "training.validation_interval_updates"
    )
    _positive_int(
        training["checkpoint_interval_updates"], "training.checkpoint_interval_updates"
    )
    if training["validation_interval_updates"] > updates:
        raise ValueError("validation interval exceeds the update budget")
    if training["checkpoint_interval_updates"] > updates:
        raise ValueError("checkpoint interval exceeds the update budget")
    if (
        training["final_validation"] is not True
        or training["final_checkpoint"] is not True
    ):
        raise ValueError("final validation and checkpoint are required")
    if training["exact_resume"] is not True:
        raise ValueError("training must require exact resume")

    optimizer = _mapping(training["optimizer"], "training.optimizer")
    _exact(
        optimizer,
        {"name", "learning_rate", "betas", "epsilon", "weight_decay", "implementation"},
        "training.optimizer",
    )
    if optimizer["name"] != "adamw":
        raise ValueError("training optimizer must be adamw")
    _finite(
        optimizer["learning_rate"], "training.optimizer.learning_rate", positive=True
    )
    betas = optimizer["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(not 0 <= _finite(beta, "training.optimizer.beta") < 1 for beta in betas)
    ):
        raise ValueError("training.optimizer.betas must contain two values in [0, 1)")
    _finite(optimizer["epsilon"], "training.optimizer.epsilon", positive=True)
    if _finite(optimizer["weight_decay"], "training.optimizer.weight_decay") < 0:
        raise ValueError("training.optimizer.weight_decay must be non-negative")
    _binding(optimizer["implementation"], "training.optimizer.implementation")

    scheduler = _mapping(training["scheduler"], "training.scheduler")
    _exact(
        scheduler,
        {"name", "total_updates", "warmup_updates", "minimum_learning_rate"},
        "training.scheduler",
    )
    if scheduler["name"] != "cosine" or scheduler["total_updates"] != updates:
        raise ValueError("cosine scheduler total must match optimizer updates")
    warmup = _positive_int(
        scheduler["warmup_updates"], "training.scheduler.warmup_updates", zero=True
    )
    if warmup >= updates:
        raise ValueError("scheduler warmup must be smaller than total updates")
    minimum_lr = _finite(
        scheduler["minimum_learning_rate"],
        "training.scheduler.minimum_learning_rate",
        positive=True,
    )
    if minimum_lr > float(optimizer["learning_rate"]):
        raise ValueError("scheduler minimum learning rate exceeds peak learning rate")

    evaluation = _mapping(plan["evaluation"], "evaluation")
    _exact(
        evaluation,
        {
            "suite",
            "development_dimensions",
            "held_out_dimensions",
            "development_before_training",
            "held_out_before_training",
            "held_out_opening_point",
            "promotion_rules",
            "rejection_rules",
        },
        "evaluation",
    )
    evaluation_binding = _binding(evaluation["suite"], "evaluation.suite")
    if evaluation_binding != gate_identities["base_evaluation_v2"][0]:
        raise ValueError(
            "selected evaluation suite does not match accepted gate evidence"
        )
    for field in ("development_dimensions", "held_out_dimensions"):
        dimensions = evaluation[field]
        if (
            not isinstance(dimensions, list)
            or set(dimensions) != REQUIRED_DIMENSIONS
            or len(dimensions) != len(REQUIRED_DIMENSIONS)
        ):
            raise ValueError(
                f"evaluation.{field} must cover every dimension exactly once"
            )
    if (
        evaluation["development_before_training"] is not True
        or evaluation["held_out_before_training"] is not False
    ):
        raise ValueError("evaluation split-opening policy mismatch")
    _string(evaluation["held_out_opening_point"], "evaluation.held_out_opening_point")
    _validate_rule_list(evaluation["promotion_rules"], "evaluation.promotion_rules")
    _validate_rule_list(evaluation["rejection_rules"], "evaluation.rejection_rules")

    _validate_runtime_envelope(
        plan["runtime_envelope"],
        updates=updates,
        input_positions=processed_records * SEQUENCE_LENGTH,
    )
    _validate_safety(plan["safety"])
    _validate_outputs(plan["outputs"])
    review = _mapping(plan["review"], "review")
    _exact(review, {"status", "reviewer", "decision_path"}, "review")
    if review != {
        "status": "pending_independent_review",
        "reviewer": "",
        "decision_path": None,
    }:
        raise ValueError("plan review state must remain pending and non-authorizing")
    if plan["training_authorized"] is not False:
        raise ValueError("training_authorized must be false")
    if _sha(plan["plan_sha256"], "plan_sha256") != plan_identity(plan):
        raise ValueError("base-pretraining plan identity mismatch")


def _validate_safety(value: object) -> None:
    safety = _mapping(value, "safety")
    _exact(safety, {"disk", "thermal", "checkpoint", "abort_on"}, "safety")
    disk = _mapping(safety["disk"], "safety.disk")
    _exact(
        disk,
        {"minimum_free_bytes", "before_launch", "before_checkpoint"},
        "safety.disk",
    )
    minimum_free = _positive_int(
        disk["minimum_free_bytes"], "safety.disk.minimum_free_bytes"
    )
    if minimum_free < 10_000_000_000:
        raise ValueError("disk safeguard must reserve at least 10 GB")
    if disk["before_launch"] is not True or disk["before_checkpoint"] is not True:
        raise ValueError("disk safeguards must run before launch and checkpoints")
    thermal = _mapping(safety["thermal"], "safety.thermal")
    _exact(
        thermal,
        {
            "required",
            "warning_celsius",
            "abort_celsius",
            "critical_celsius",
            "consecutive_abort_readings",
        },
        "safety.thermal",
    )
    warning = _finite(thermal["warning_celsius"], "safety.thermal.warning_celsius")
    abort = _finite(thermal["abort_celsius"], "safety.thermal.abort_celsius")
    critical = _finite(thermal["critical_celsius"], "safety.thermal.critical_celsius")
    if (
        thermal["required"] is not True
        or not 0 < warning < abort <= 88
        or not abort < critical <= 95
    ):
        raise ValueError("thermal safeguards must have required ordered thresholds")
    _positive_int(
        thermal["consecutive_abort_readings"],
        "safety.thermal.consecutive_abort_readings",
    )
    checkpoint = _mapping(safety["checkpoint"], "safety.checkpoint")
    _exact(
        checkpoint,
        {"atomic", "retention_count", "integrity_sidecars"},
        "safety.checkpoint",
    )
    if checkpoint["atomic"] is not True or checkpoint["integrity_sidecars"] is not True:
        raise ValueError("checkpoint atomicity and integrity sidecars are required")
    _positive_int(checkpoint["retention_count"], "safety.checkpoint.retention_count")
    abort_on = safety["abort_on"]
    required = {
        "cuda_oom",
        "nonfinite",
        "checkpoint_failure",
        "validation_failure",
        "resume_divergence",
        "thermal_limit",
        "disk_limit",
    }
    if (
        not isinstance(abort_on, list)
        or set(abort_on) != required
        or len(abort_on) != len(required)
    ):
        raise ValueError("safety.abort_on must contain every required hard stop")


def _validate_outputs(value: object) -> None:
    outputs = _mapping(value, "outputs")
    _exact(
        outputs,
        {"checkpoint_directory", "log_directory", "result_directory", "must_be_absent"},
        "outputs",
    )
    paths = [
        _safe_path(outputs[field], f"outputs.{field}")
        for field in ("checkpoint_directory", "log_directory", "result_directory")
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("output directories must be distinct")
    prefixes = {
        "checkpoint_directory": "checkpoints/vasu_140m/",
        "log_directory": "logs/vasu_140m/",
        "result_directory": "evaluation/results/vasu_140m/",
    }
    for field, path in zip(
        ("checkpoint_directory", "log_directory", "result_directory"),
        paths,
        strict=True,
    ):
        if not path.startswith(prefixes[field]):
            raise ValueError(f"outputs.{field} is outside its isolated root")
    path_parts = [Path(path).parts for path in paths]
    for index, left in enumerate(path_parts):
        for right in path_parts[index + 1 :]:
            common = min(len(left), len(right))
            if left[:common] == right[:common]:
                raise ValueError("output directories must not overlap")
    if outputs["must_be_absent"] is not True:
        raise ValueError("all output directories must be absent before launch")


def _validate_runtime_envelope(
    value: object, *, updates: int, input_positions: int
) -> None:
    runtime = _mapping(value, "runtime_envelope")
    _exact(
        runtime,
        {
            "minimum_input_positions_per_second",
            "maximum_input_positions_per_second",
            "maximum_wall_time_seconds",
            "maximum_optimizer_updates",
            "maximum_input_positions",
            "stop_at_budget",
        },
        "runtime_envelope",
    )
    minimum = _finite(
        runtime["minimum_input_positions_per_second"],
        "runtime_envelope.minimum_input_positions_per_second",
        positive=True,
    )
    maximum = _finite(
        runtime["maximum_input_positions_per_second"],
        "runtime_envelope.maximum_input_positions_per_second",
        positive=True,
    )
    if maximum < minimum:
        raise ValueError("runtime throughput envelope is reversed")
    wall_time = _positive_int(
        runtime["maximum_wall_time_seconds"],
        "runtime_envelope.maximum_wall_time_seconds",
    )
    if wall_time < math.ceil(input_positions / minimum):
        raise ValueError("runtime wall-time limit is below the slow-bound estimate")
    if runtime["maximum_optimizer_updates"] != updates:
        raise ValueError("runtime optimizer-update cap does not match training")
    if runtime["maximum_input_positions"] != input_positions:
        raise ValueError("runtime input-position cap does not match training")
    if runtime["stop_at_budget"] is not True:
        raise ValueError("runtime must stop exactly at the frozen budget")


def validate_base_pretraining_plan_files(
    plan: Mapping[str, object], repository_root: Path, *, runtime_commit: str
) -> None:
    """Verify every bound file and output absence without creating runtime state."""

    validate_base_pretraining_plan(plan)
    if _commit(runtime_commit, "runtime_commit") != plan["repository_commit"]:
        raise ValueError("runtime commit does not match the plan")
    root = repository_root.resolve()
    bindings: list[tuple[str, str, str]] = []
    bindings.append((*_binding(plan["tokenizer"], "tokenizer"), "tokenizer"))
    for name, raw in _mapping(plan["gate_evidence"], "gate_evidence").items():
        gate = _mapping(raw, f"gate {name}")
        bindings.append(
            (
                *_binding(gate["artifact"], f"gate {name}.artifact"),
                f"gate {name} artifact",
            )
        )
        bindings.append(
            (
                *_binding(gate["decision"], f"gate {name}.decision"),
                f"gate {name} decision",
            )
        )
    initialization = _mapping(plan["initialization"], "initialization")
    bindings.append(
        (
            *_binding(
                initialization["implementation"], "initialization.implementation"
            ),
            "initialization implementation",
        )
    )
    data = _mapping(plan["data"], "data")
    bindings.append((*_binding(data["release"], "data.release"), "data release"))
    schedule = _mapping(data["schedule"], "data.schedule")
    bindings.append(
        (
            _safe_path(schedule["path"], "data.schedule.path"),
            _sha(schedule["sha256"], "data.schedule.sha256"),
            "data schedule",
        )
    )
    training = _mapping(plan["training"], "training")
    optimizer = _mapping(training["optimizer"], "training.optimizer")
    bindings.append(
        (
            *_binding(optimizer["implementation"], "training.optimizer.implementation"),
            "optimizer implementation",
        )
    )
    evaluation = _mapping(plan["evaluation"], "evaluation")
    bindings.append(
        (*_binding(evaluation["suite"], "evaluation.suite"), "evaluation suite")
    )
    for relative, expected, label in bindings:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{label} file is missing or escapes the repository")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} file identity mismatch")
    outputs = _mapping(plan["outputs"], "outputs")
    for field in ("checkpoint_directory", "log_directory", "result_directory"):
        path = (root / _safe_path(outputs[field], f"outputs.{field}")).resolve()
        if not path.is_relative_to(root) or path.exists():
            raise ValueError(f"outputs.{field} must be absent inside the repository")
