"""CPU-only helpers for safe VASU checkpoint orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping, Sequence

import torch

from vasu.config import get_vasu_60m_config
from vasu.model.model import VASUModel


REQUIRED_CHECKPOINT_KEYS = frozenset(
    {"epoch", "global_step", "loss", "model", "optimizer", "scheduler"}
)
CHECKPOINT_PATTERNS = (
    re.compile(r"^block_final_step_(\d+)\.pt$"),
    re.compile(r"^step_(\d+)\.pt$"),
    re.compile(r"^thermal_stop_step_(\d+)\.pt$"),
)


@dataclass(frozen=True)
class CheckpointInspection:
    valid: bool
    path: str
    global_step: int | None
    model_config: str | None
    has_model_state: bool
    has_optimizer_state: bool
    has_scheduler_state: bool
    finite_tensors: bool | None
    sha256: str | None
    filename_step: int | None
    filename_step_matches: bool | None
    checkpoint_kind: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _checkpoint_filename_details(path: Path) -> tuple[str | None, int | None]:
    names = ("block_final", "periodic", "thermal_stop")
    for name, pattern in zip(names, CHECKPOINT_PATTERNS, strict=True):
        match = pattern.fullmatch(path.name)
        if match:
            return name, int(match.group(1))
    return None, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_tensors_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            return bool(torch.isfinite(value).all())
        return True
    if isinstance(value, Mapping):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_tensors_finite(item) for item in value)
    return True


def _validate_vasu_60m_model_state(model_state: Mapping[str, Any]) -> None:
    # Meta construction validates every state-dict key and tensor shape without
    # allocating a second 60M-parameter CPU model.
    with torch.device("meta"):
        expected_state = VASUModel(get_vasu_60m_config()).state_dict()
    if set(model_state) != set(expected_state):
        missing = sorted(set(expected_state) - set(model_state))
        unexpected = sorted(set(model_state) - set(expected_state))
        raise ValueError(
            "model state does not match VASU-60M keys; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for key, expected in expected_state.items():
        actual = model_state[key]
        if not torch.is_tensor(actual):
            raise ValueError(f"model state entry is not a tensor: {key}")
        if actual.shape != expected.shape:
            raise ValueError(
                f"VASU-60M tensor shape mismatch for {key}: "
                f"found {tuple(actual.shape)}, expected {tuple(expected.shape)}"
            )


def inspect_checkpoint(
    path: str | Path,
    *,
    verify_finite: bool = True,
    calculate_hash: bool = True,
) -> CheckpointInspection:
    """Validate one checkpoint on CPU without initializing CUDA."""

    checkpoint_path = Path(path).resolve()
    kind, filename_step = _checkpoint_filename_details(checkpoint_path)
    base = {
        "path": str(checkpoint_path),
        "filename_step": filename_step,
        "checkpoint_kind": kind,
    }
    if checkpoint_path.suffix != ".pt" or checkpoint_path.name.endswith(".tmp"):
        return CheckpointInspection(
            valid=False,
            global_step=None,
            model_config=None,
            has_model_state=False,
            has_optimizer_state=False,
            has_scheduler_state=False,
            finite_tensors=None,
            sha256=None,
            filename_step_matches=None,
            error="candidate must be a non-temporary .pt checkpoint",
            **base,
        )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint payload is not a mapping")
        missing = REQUIRED_CHECKPOINT_KEYS - set(checkpoint)
        if missing:
            raise ValueError(f"missing required checkpoint keys: {sorted(missing)}")
        global_step = checkpoint["global_step"]
        if isinstance(global_step, bool) or not isinstance(global_step, int):
            raise ValueError("global_step must be an integer")
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        model_state = checkpoint["model"]
        optimizer_state = checkpoint["optimizer"]
        scheduler_state = checkpoint["scheduler"]
        if not isinstance(model_state, Mapping) or not model_state:
            raise ValueError("model state is missing or empty")
        if not isinstance(optimizer_state, Mapping) or not optimizer_state:
            raise ValueError("optimizer state is missing or empty")
        if not isinstance(scheduler_state, Mapping) or not scheduler_state:
            raise ValueError("scheduler state is missing or empty")
        _validate_vasu_60m_model_state(model_state)
        finite = _all_tensors_finite(checkpoint) if verify_finite else None
        if finite is False:
            raise ValueError("checkpoint contains non-finite floating tensors")
        digest = _sha256(checkpoint_path) if calculate_hash else None
        return CheckpointInspection(
            valid=True,
            global_step=global_step,
            model_config="vasu_60m",
            has_model_state=True,
            has_optimizer_state=True,
            has_scheduler_state=True,
            finite_tensors=finite,
            sha256=digest,
            filename_step_matches=(
                filename_step == global_step if filename_step is not None else None
            ),
            error=None,
            **base,
        )
    except Exception as error:  # Invalid candidates must be reportable, not fatal.
        return CheckpointInspection(
            valid=False,
            global_step=None,
            model_config=None,
            has_model_state=False,
            has_optimizer_state=False,
            has_scheduler_state=False,
            finite_tensors=None,
            sha256=None,
            filename_step_matches=None,
            error=f"{type(error).__name__}: {error}",
            **base,
        )


def find_latest_valid_checkpoint(
    checkpoint_dir: str | Path,
    *,
    verify_finite: bool = True,
    calculate_hash: bool = True,
) -> CheckpointInspection:
    """Return the highest internally recorded valid checkpoint step."""

    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {directory}")
    candidates = [
        path
        for path in directory.glob("*.pt")
        if not path.name.endswith(".tmp")
    ]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint candidates found in {directory}")

    # First inspect metadata and VASU-60M shapes using mmap. Only the selected
    # checkpoint receives the expensive finite scan and SHA-256 pass.
    valid = []
    for candidate in candidates:
        inspection = inspect_checkpoint(
            candidate,
            verify_finite=False,
            calculate_hash=False,
        )
        if inspection.valid:
            valid.append(inspection)
    if not valid:
        raise RuntimeError(f"no valid VASU-60M checkpoints found in {directory}")

    kind_priority = {"block_final": 3, "periodic": 2, "thermal_stop": 1, None: 0}
    selected = max(
        valid,
        key=lambda item: (
            item.global_step if item.global_step is not None else -1,
            kind_priority[item.checkpoint_kind],
        ),
    )
    final = inspect_checkpoint(
        selected.path,
        verify_finite=verify_finite,
        calculate_hash=calculate_hash,
    )
    if not final.valid:
        raise RuntimeError(f"selected checkpoint failed validation: {final.error}")
    return final


def validate_progress(
    previous_step: int,
    new_step: int,
    target_step: int,
    *,
    block_steps: int = 200,
) -> int:
    """Validate one bounded block transition and return its step delta."""

    if previous_step > target_step:
        raise ValueError("starting global step is above the wrapper target")
    if previous_step == target_step:
        if new_step != target_step:
            raise ValueError("target was already reached but progress changed")
        return 0
    delta = new_step - previous_step
    if delta <= 0:
        raise ValueError("global step did not increase")
    if delta > block_steps:
        raise ValueError(f"global step increased by more than {block_steps}")
    if new_step > target_step:
        raise ValueError("global step exceeded the wrapper target")
    expected_delta = min(block_steps, target_step - previous_step)
    if delta != expected_delta:
        raise ValueError(
            f"unexpected progress delta {delta}; expected {expected_delta}"
        )
    return delta


def validate_process_exit(exit_code: int) -> None:
    if exit_code != 0:
        raise RuntimeError(f"training process exited with status {exit_code}")


def require_free_disk(free_gib: float, minimum_gib: float) -> None:
    if free_gib < minimum_gib:
        raise RuntimeError(
            f"free disk {free_gib:.2f} GiB is below {minimum_gib:.2f} GiB"
        )


def should_launch_next_block(
    *,
    current_step: int,
    target_step: int,
    interrupted: bool = False,
    thermal_stop: bool = False,
    auto_resume_after_thermal: bool = False,
) -> bool:
    if interrupted or current_step >= target_step:
        return False
    if thermal_stop and not auto_resume_after_thermal:
        return False
    return True


def parse_runner_output(output: str) -> dict[str, Any]:
    """Conservatively extract optional block metrics from runner text."""

    patterns: dict[str, tuple[str, type]] = {
        "starting_step": (r"Starting global step:\s*(\d+)", int),
        "final_step": (r"Final global step:\s*(\d+)", int),
        "train_loss": (r"Latest loss:\s*([0-9.eE+-]+)", float),
        "validation_loss": (r"Validation loss:\s*([0-9.eE+-]+)", float),
        "maximum_temperature": (r"Maximum GPU temperature:\s*(\d+)\s*C", int),
        "peak_cuda_memory_mib": (
            r"CUDA peak memory:\s*([0-9.eE+-]+)\s*MiB",
            float,
        ),
        "free_disk_after_gib": (
            r"Free disk after training:\s*([0-9.eE+-]+)\s*GiB",
            float,
        ),
    }
    result: dict[str, Any] = {}
    for name, (pattern, converter) in patterns.items():
        matches = re.findall(pattern, output)
        result[name] = converter(matches[-1]) if matches else None
    completed = re.findall(r"Block completed:\s*(True|False)", output)
    thermal = re.findall(r"Thermal stop occurred:\s*(True|False)", output)
    result["block_completed"] = (
        completed[-1] == "True" if completed else None
    )
    result["thermal_stop"] = thermal[-1] == "True" if thermal else None
    return result


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a durable UTF-8 JSONL line."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def preserve_milestone(
    source: str | Path,
    destination: str | Path,
    *,
    expected_step: int,
) -> dict[str, Any]:
    """Atomically preserve a validated checkpoint without overwriting history."""

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(f"milestone already exists: {destination_path}")
    inspection = inspect_checkpoint(source_path)
    if not inspection.valid or inspection.global_step != expected_step:
        raise ValueError(
            f"source is not a valid step-{expected_step} VASU-60M checkpoint"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source_path, temporary)
        if _sha256(temporary) != inspection.sha256:
            raise RuntimeError("temporary milestone hash does not match source")
        os.replace(temporary, destination_path)
        destination_inspection = inspect_checkpoint(destination_path)
        if not destination_inspection.valid:
            raise RuntimeError("preserved milestone failed validation")
        if destination_inspection.sha256 != inspection.sha256:
            raise RuntimeError("preserved milestone hash does not match source")
        return destination_inspection.to_dict()
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_disk_gib(path: str | Path) -> float:
    return shutil.disk_usage(Path(path)).free / (1024**3)


def conflicting_commands(
    processes: Iterable[tuple[int, str]],
    *,
    excluded_pids: Sequence[int] = (),
) -> list[dict[str, Any]]:
    """Return process records whose command line looks like VASU training."""

    excluded = set(excluded_pids)
    terms = ("train_vasu", "fineweb", "alpaca", "ultrachat")
    conflicts = []
    for pid, command_line in processes:
        lowered = command_line.lower()
        if pid not in excluded and any(term in lowered for term in terms):
            conflicts.append({"pid": pid, "command_line": command_line})
    return conflicts
