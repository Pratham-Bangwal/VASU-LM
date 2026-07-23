"""Validated loading for capability suites and legacy checkpoint registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vasu.data.prompt_templates import get_prompt_formatter

from .schemas import CapabilityTask, CheckpointEntry, HUMAN, HEURISTIC, OBJECTIVE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def load_checkpoint_registry(path: Path) -> dict[str, CheckpointEntry]:
    """Load legacy-compatible checkpoint entries with explicit defaults."""

    raw = _read_json(path).get("checkpoints")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint registry requires a 'checkpoints' object.")
    entries: dict[str, CheckpointEntry] = {}
    for identifier, entry in raw.items():
        if isinstance(entry, str):
            entry = {"path": entry, "model_config": "vasu_31m", "prompt_format": "alpaca"}
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"Checkpoint {identifier!r} has no valid path.")
        model_config = entry.get("model_config", "vasu_31m")
        prompt_format = entry.get("prompt_format", "alpaca")
        if model_config not in {"vasu_31m", "vasu_60m"}:
            raise ValueError(f"Checkpoint {identifier!r} has unknown model config.")
        get_prompt_formatter(prompt_format)
        entries[identifier] = CheckpointEntry(identifier, entry["path"], model_config, prompt_format)
    return entries


def load_suite(path: Path) -> tuple[dict[str, Any], list[CapabilityTask]]:
    """Load a suite while keeping objective, heuristic, and human tasks distinct."""

    raw = _read_json(path)
    if not isinstance(raw.get("suite_version"), str) or not raw["suite_version"]:
        raise ValueError("Suite requires a non-empty suite_version.")
    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("Suite requires a non-empty tasks list.")
    tasks: list[CapabilityTask] = []
    seen: set[str] = set()
    for item in tasks_raw:
        required = {"id", "category", "prompt", "metric_kind", "scoring", "provenance"}
        if not isinstance(item, dict) or required - set(item):
            raise ValueError("Each suite task needs id/category/prompt/metric_kind/scoring/provenance.")
        identifier = item["id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Suite task IDs must be unique non-empty strings.")
        if item["metric_kind"] not in {OBJECTIVE, HEURISTIC, HUMAN}:
            raise ValueError(f"Task {identifier!r} has an unsupported metric_kind.")
        if not isinstance(item["provenance"], dict) or not item["provenance"].get("source"):
            raise ValueError(f"Task {identifier!r} has no provenance source.")
        if not isinstance(item["scoring"], dict) or not item["scoring"].get("rule"):
            raise ValueError(f"Task {identifier!r} has no scoring rule.")
        seen.add(identifier)
        tasks.append(CapabilityTask(identifier, item["category"], item["prompt"], item["metric_kind"], item["scoring"], item["provenance"], raw["suite_version"]))
    _validate_promotion_gates(raw.get("promotion_gates"), tasks)
    return raw, tasks


def _validate_promotion_gates(
    gates: Any, tasks: list[CapabilityTask]
) -> None:
    """Validate explicit promotion scopes before a suite can be executed."""

    if gates is None:
        return
    if not isinstance(gates, dict):
        raise ValueError("Suite requires a promotion_gates object.")
    objective_categories = {
        task.category for task in tasks if task.metric_kind == OBJECTIVE
    }
    required = {"continued_pretraining", "instruction", "conversation"}
    if set(gates) != required:
        raise ValueError("promotion_gates must define continued_pretraining, instruction, and conversation.")
    required_fields = {
        "continued_pretraining": {
            "targeted_categories",
            "general_categories",
            "targeted_objective_delta_minimum",
            "general_objective_regression_maximum",
            "human_review_required",
        },
        "instruction": {
            "formatting_categories",
            "factual_categories",
            "formatting_regression_maximum",
            "factual_regression_maximum",
            "human_review_required",
        },
        "conversation": {
            "objective_categories",
            "objective_regression_maximum",
            "human_review_required",
        },
    }
    for name, gate in gates.items():
        if not isinstance(gate, dict):
            raise ValueError(f"Promotion gate {name!r} must be an object.")
        missing = required_fields[name] - set(gate)
        if missing:
            raise ValueError(
                f"Promotion gate {name!r} is missing required fields: "
                f"{sorted(missing)}"
            )
        has_other_threshold = "other_objective_regression_maximum" in gate
        has_other_categories = "other_objective_categories" in gate
        if has_other_threshold != has_other_categories:
            raise ValueError(
                "Instruction's optional other-objective gate requires both "
                "other_objective_categories and other_objective_regression_maximum."
            )
        for key, value in gate.items():
            if key.endswith(("_minimum", "_maximum")) and (
                not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"Promotion gate {name!r} has invalid threshold {key!r}.")
        for key, value in gate.items():
            if key.endswith("_categories"):
                if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
                    raise ValueError(f"Promotion gate {name!r} has invalid {key!r}.")
                unknown = set(value) - objective_categories
                if unknown:
                    raise ValueError(f"Promotion gate {name!r} names unknown/non-objective categories: {sorted(unknown)}")
        if not isinstance(gate.get("human_review_required"), bool):
            raise ValueError(f"Promotion gate {name!r} requires boolean human_review_required.")
