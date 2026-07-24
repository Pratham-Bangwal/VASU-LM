"""Production safeguards and domain validation for capability-CPT runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.verified_arithmetic import (
    evaluate_records,
    generate_answer,
    summarize_results,
)
from vasu.training.checkpoint import save_checkpoint
from vasu.training.losses import language_model_loss
from vasu.training.resume_state import capture_rng_state, restore_rng_state
from vasu.training.trainer import Trainer


CAPABILITY_RUNTIME_VERSION = "vasu_capability_runtime_v1"
CAPABILITY_CHECKPOINT_KEYS = frozenset(
    {
        "epoch",
        "global_step",
        "model",
        "optimizer",
        "scheduler",
        "training_progress",
        "capability_identity",
        "capability_state",
    }
)
DOMAIN_BEST_NAMES = (
    "best_fineweb.pt",
    "best_wikimedia.pt",
    "best_arithmetic.pt",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: Path = Path(".")) -> dict[str, Any]:
    """Return the exact commit and porcelain status without mutating Git."""

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Git state is unavailable") from error
    return {"commit": commit, "clean": not status, "changes": status}


def validation_configuration_hash(config: Mapping[str, Any]) -> str:
    payload = {
        "validation_interval": config["validation_interval"],
        "validation": config["validation"],
        "best_checkpoint_policy": config["best_checkpoint_policy"],
        "abort_policy": config["abort_policy"],
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def build_capability_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the compact immutable identity required for exact resume."""

    identity = {
        "runtime_version": CAPABILITY_RUNTIME_VERSION,
        "experiment_id": config["experiment_id"],
        "parent_checkpoint": config["parent_checkpoint"],
        "model_configuration": config["model_configuration"],
        "tokenizer": config["tokenizer"],
        "resolved_mixture_manifest_sha256": config[
            "resolved_mixture_manifest_sha256"
        ],
        "schedule_sha256": config["expected_schedule_sha256"],
        "source_hashes": config["expected_source_hashes"],
        "validation_configuration_sha256": validation_configuration_hash(config),
        "optimizer_backend": config["optimizer_backend"],
        "scheduler": config["scheduler"],
        "batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "sequence_length": config["sequence_length"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
    }
    identity["sha256"] = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    return identity


class FlatTokenValidationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read-only uint16 validation stream with the standard target shift."""

    def __init__(
        self,
        path: Path,
        sequence_length: int,
        *,
        expected_sha256: str,
        maximum_records: int | None = None,
    ) -> None:
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"validation token SHA-256 mismatch: {path}")
        self.path = path
        self.sequence_length = sequence_length
        count = path.stat().st_size // np.dtype(np.uint16).itemsize
        available = max(0, (count - 1) // sequence_length)
        self.record_count = (
            min(available, maximum_records)
            if maximum_records is not None
            else available
        )
        if self.record_count < 1:
            raise ValueError("validation token stream has no complete records")
        self._tokens: np.memmap | None = None

    def _open(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return self.record_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        row = np.asarray(
            self._open()[start : start + self.sequence_length + 1],
            dtype=np.int64,
        )
        return torch.from_numpy(row[:-1].copy()), torch.from_numpy(row[1:].copy())


@dataclass
class ThermalState:
    maximum_temperature: int | None = None
    consecutive_abort_readings: int = 0
    readings: list[dict[str, int]] = field(default_factory=list)


class ThermalMonitor:
    """Bounded nvidia-smi monitor with sustained and critical aborts."""

    def __init__(
        self,
        *,
        warning_celsius: int,
        abort_celsius: int,
        critical_celsius: int,
        consecutive_abort_readings: int,
        reader: Callable[[], int | None] | None = None,
    ) -> None:
        if not warning_celsius < abort_celsius < critical_celsius:
            raise ValueError("thermal thresholds must be strictly increasing")
        if consecutive_abort_readings < 1:
            raise ValueError("consecutive thermal readings must be positive")
        self.warning = warning_celsius
        self.abort = abort_celsius
        self.critical = critical_celsius
        self.required = consecutive_abort_readings
        self.reader = reader or self._nvidia_smi_temperature
        self.state = ThermalState()

    @staticmethod
    def _nvidia_smi_temperature() -> int | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return int(result.stdout.strip().splitlines()[0])
        except (
            FileNotFoundError,
            subprocess.SubprocessError,
            ValueError,
            IndexError,
        ):
            return None

    def read(self, step: int) -> dict[str, Any]:
        temperature = self.reader()
        if temperature is None:
            return {"available": False, "warning": False, "abort": False}
        self.state.maximum_temperature = max(
            temperature,
            self.state.maximum_temperature
            if self.state.maximum_temperature is not None
            else temperature,
        )
        self.state.readings.append({"step": step, "celsius": temperature})
        if temperature >= self.abort:
            self.state.consecutive_abort_readings += 1
        else:
            self.state.consecutive_abort_readings = 0
        abort = (
            temperature >= self.critical
            or self.state.consecutive_abort_readings >= self.required
        )
        return {
            "available": True,
            "celsius": temperature,
            "warning": temperature >= self.warning,
            "abort": abort,
        }


def checkpoint_disk_requirement_bytes(config: Mapping[str, Any]) -> int:
    """Estimate retained state, atomic duplication, and configured margin."""

    policy = config["disk_safety"]
    boundary = int(policy["estimated_boundary_checkpoint_bytes"])
    mid = int(policy["estimated_mid_accumulation_checkpoint_bytes"])
    retained = (
        int(config["checkpoint_retention"]["periodic_keep"]) * boundary
        + len(set(config["checkpoint_retention"]["milestone_steps"])) * boundary
        + len(DOMAIN_BEST_NAMES) * boundary
        + 2 * boundary  # latest and final
        + mid  # largest sibling temporary file
    )
    margin = int(float(policy["minimum_free_space_margin_gib"]) * 1024**3)
    return retained + margin


def require_disk_space(config: Mapping[str, Any], destination: Path) -> int:
    required = checkpoint_disk_requirement_bytes(config)
    available = shutil.disk_usage(destination.parent).free
    if available < required:
        raise RuntimeError(
            "insufficient checkpoint disk space: "
            f"available={available}, required={required}"
        )
    return available


def inspect_capability_checkpoint(
    path: Path,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a named resume checkpoint on CPU before model allocation."""

    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    if path.name.endswith(".tmp") or path.suffix != ".pt":
        raise ValueError("resume checkpoint must be a completed .pt file")
    if path.stat().st_size < 1024:
        raise ValueError("resume checkpoint is zero-byte or obviously truncated")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except Exception as error:
        raise ValueError(f"resume checkpoint cannot be loaded: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("resume checkpoint payload is not a mapping")
    missing = CAPABILITY_CHECKPOINT_KEYS - set(payload)
    if missing:
        raise ValueError(f"resume checkpoint missing keys: {sorted(missing)}")
    if payload["capability_identity"] != expected_identity:
        raise ValueError("resume checkpoint capability identity mismatch")
    model_state = payload.get("model")
    expected_shapes = {
        "embedding.embedding.weight": (32_000, 512),
        "lm_head.weight": (32_000, 512),
        "blocks.0.attention.q_proj.weight": (512, 512),
        "blocks.9.mlp.w1.weight": (2_048, 512),
    }
    if not isinstance(model_state, Mapping):
        raise ValueError("resume checkpoint model state is invalid")
    for key, shape in expected_shapes.items():
        value = model_state.get(key)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"resume checkpoint architecture mismatch at {key}")
    state = payload["capability_state"]
    if not isinstance(state, Mapping):
        raise ValueError("resume checkpoint capability_state is invalid")
    return {
        "global_step": int(payload["global_step"]),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "capability_state": dict(state),
    }


def write_checkpoint_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(path.suffix + ".json")
    payload = {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)
    return sidecar


def apply_checkpoint_retention(
    directory: Path,
    *,
    keep_periodic: int,
    milestones: Sequence[int],
    dry_run: bool = False,
) -> list[Path]:
    """Remove only old validated periodic checkpoints."""

    periodic = []
    for path in directory.glob("step_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        periodic.append((step, path))
    protected_steps = set(int(step) for step in milestones)
    candidates = [
        item for item in sorted(periodic, reverse=True) if item[0] not in protected_steps
    ]
    keep_paths = {path for _, path in candidates[:keep_periodic]}
    removed = []
    for _, path in candidates[keep_periodic:]:
        if path in keep_paths:
            continue
        removed.append(path)
        if not dry_run:
            path.unlink()
            path.with_suffix(path.suffix + ".json").unlink(missing_ok=True)
    return removed


@torch.inference_mode()
def validation_loss(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> float:
    total = 0.0
    batches = 0
    for batch in loader:
        inputs, targets = batch[:2]
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            loss = language_model_loss(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("validation loss is non-finite")
        total += float(loss.item())
        batches += 1
    if not batches:
        raise RuntimeError("validation loader is empty")
    return total / batches


class CapabilityTrainer(Trainer):
    """Opt-in Trainer variant with durable multi-domain validation events."""

    def __init__(
        self,
        *args: Any,
        capability_config: Mapping[str, Any],
        capability_identity: Mapping[str, Any],
        validation_loaders: Mapping[str, DataLoader[Any]],
        arithmetic_records: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        self.capability_config = dict(capability_config)
        self.capability_identity = dict(capability_identity)
        self.validation_loaders = dict(validation_loaders)
        self.arithmetic_records = list(arithmetic_records)
        self.validation_events: list[dict[str, Any]] = []
        self.best_metrics = {
            "fineweb": math.inf,
            "wikimedia": math.inf,
            "arithmetic": -math.inf,
            "arithmetic_malformed": math.inf,
            "arithmetic_duration": math.inf,
        }
        self.optimizer_skips_total = 0
        self.optimizer_skips_consecutive = 0
        thermal = self.capability_config["thermal_safety"]
        self.thermal_monitor = ThermalMonitor(
            warning_celsius=int(thermal["warning_celsius"]),
            abort_celsius=int(thermal["abort_celsius"]),
            critical_celsius=int(thermal["critical_celsius"]),
            consecutive_abort_readings=int(
                thermal["consecutive_abort_readings"]
            ),
        )
        self.abort_reason: str | None = None
        super().__init__(*args, **kwargs)

    def _load_checkpoint(self) -> None:
        super()._load_checkpoint()
        path = Path(self.config.checkpoint_path)
        if not path.exists():
            return
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if payload.get("capability_identity") != self.capability_identity:
            raise ValueError("checkpoint capability identity mismatch")
        state = payload.get("capability_state")
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint has no capability validation state")
        self.validation_events = list(state.get("validation_events", []))
        self.best_metrics.update(state.get("best_metrics", {}))
        self.optimizer_skips_total = int(state.get("optimizer_skips_total", 0))
        self.optimizer_skips_consecutive = int(
            state.get("optimizer_skips_consecutive", 0)
        )

    def _capability_state(self) -> dict[str, Any]:
        return {
            "validation_events": self.validation_events,
            "completed_validation_steps": [
                int(event["experiment_step"]) for event in self.validation_events
            ],
            "best_metrics": self.best_metrics,
            "optimizer_skips_total": self.optimizer_skips_total,
            "optimizer_skips_consecutive": self.optimizer_skips_consecutive,
            "retention": self.capability_config["checkpoint_retention"],
            "thermal": asdict(self.thermal_monitor.state),
            "abort_reason": self.abort_reason,
        }

    def save_training_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int | None = None,
        loss: float | None = None,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        require_disk_space(self.capability_config, destination)
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.start_epoch if epoch is None else epoch,
            loss=loss,
            path=destination,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            training_progress=self._training_progress(),
            capability_identity=self.capability_identity,
            capability_state=self._capability_state(),
            verify_after_write=True,
            fsync=True,
        )
        write_checkpoint_sidecar(destination)

    def _run_validation_event(self) -> dict[str, Any]:
        if any(
            int(event["experiment_step"]) == self.global_step
            for event in self.validation_events
        ):
            return next(
                event
                for event in self.validation_events
                if int(event["experiment_step"]) == self.global_step
            )
        training_mode = self.model.training
        rng_state = capture_rng_state()
        self.model.eval()
        durations: dict[str, float] = {}
        try:
            started = time.perf_counter()
            fineweb = validation_loss(
                self.model, self.validation_loaders["fineweb"], self.device
            )
            durations["fineweb"] = time.perf_counter() - started
            started = time.perf_counter()
            wikimedia = validation_loss(
                self.model, self.validation_loaders["wikimedia"], self.device
            )
            durations["wikimedia"] = time.perf_counter() - started
            started = time.perf_counter()
            arithmetic = evaluate_records(
                self.arithmetic_records,
                lambda record: generate_answer(
                    self.model,
                    self.tokenizer,
                    record,
                    self.device,
                    max_new_tokens=int(
                        self.capability_config["validation"]["arithmetic_proxy"][
                            "max_new_tokens"
                        ]
                    ),
                ),
            )
            arithmetic_summary = summarize_results(arithmetic)["overall"]
            durations["arithmetic"] = time.perf_counter() - started
        finally:
            restore_rng_state(rng_state)
            self.model.train(training_mode)
        event = {
            "experiment_step": self.global_step,
            "fineweb_loss": fineweb,
            "wikimedia_loss": wikimedia,
            "arithmetic_exact_accuracy": arithmetic_summary["exact_accuracy"],
            "arithmetic_malformed_rate": arithmetic_summary["malformed_rate"],
            "arithmetic_unanswered_rate": arithmetic_summary["unanswered_rate"],
            "arithmetic_generation": {
                "mode": "greedy",
                "do_sample": False,
                "max_new_tokens": self.capability_config["validation"][
                    "arithmetic_proxy"
                ]["max_new_tokens"],
                "record_count": len(self.arithmetic_records),
                "selection": self.capability_config["validation"][
                    "arithmetic_proxy"
                ]["selection"],
            },
            "duration_seconds": durations,
        }
        self.validation_events.append(event)
        return event

    def _save_best_checkpoints(self, event: Mapping[str, Any]) -> None:
        directory = Path(self.config.checkpoint_dir)
        fineweb = float(event["fineweb_loss"])
        wikimedia = float(event["wikimedia_loss"])
        arithmetic = float(event["arithmetic_exact_accuracy"])
        malformed = float(event["arithmetic_malformed_rate"])
        duration = float(event["duration_seconds"]["arithmetic"])
        if fineweb < self.best_metrics["fineweb"]:
            self.best_metrics["fineweb"] = fineweb
            self.save_training_checkpoint(directory / "best_fineweb.pt", loss=fineweb)
        if wikimedia < self.best_metrics["wikimedia"]:
            self.best_metrics["wikimedia"] = wikimedia
            self.save_training_checkpoint(
                directory / "best_wikimedia.pt", loss=wikimedia
            )
        arithmetic_key = (arithmetic, -malformed, -duration)
        best_key = (
            self.best_metrics["arithmetic"],
            -self.best_metrics["arithmetic_malformed"],
            -self.best_metrics["arithmetic_duration"],
        )
        if arithmetic_key > best_key:
            self.best_metrics.update(
                {
                    "arithmetic": arithmetic,
                    "arithmetic_malformed": malformed,
                    "arithmetic_duration": duration,
                }
            )
            self.save_training_checkpoint(
                directory / "best_arithmetic.pt", loss=1.0 - arithmetic
            )

    def _apply_abort_guards(self, event: Mapping[str, Any]) -> None:
        policy = self.capability_config["abort_policy"]
        fineweb = float(event["fineweb_loss"])
        wikimedia = float(event["wikimedia_loss"])
        fineweb_limit = float(policy["fineweb_baseline_loss"]) * (
            1.0 + float(policy["fineweb_max_relative_regression"])
        )
        wiki_limit = float(policy["wikimedia_baseline_loss"]) * (
            1.0 + float(policy["wikimedia_catastrophic_relative_regression"])
        )
        if fineweb > fineweb_limit:
            self.abort_reason = "fineweb_validation_guardrail"
            raise RuntimeError(
                f"FineWeb validation guardrail exceeded: {fineweb} > {fineweb_limit}"
            )
        if wikimedia > wiki_limit:
            self.abort_reason = "wikimedia_catastrophic_regression"
            raise RuntimeError(
                f"Wikimedia catastrophic guardrail exceeded: {wikimedia} > {wiki_limit}"
            )

    def _optimizer_step(self) -> bool:
        succeeded = super()._optimizer_step()
        if not succeeded:
            self.optimizer_skips_total += 1
            self.optimizer_skips_consecutive += 1
            policy = self.capability_config["abort_policy"]
            if (
                self.optimizer_skips_total
                >= int(policy["maximum_optimizer_skips_total"])
                or self.optimizer_skips_consecutive
                >= int(policy["maximum_optimizer_skips_consecutive"])
            ):
                self.abort_reason = "repeated_optimizer_skips"
                raise RuntimeError("optimizer skip threshold exceeded")
            return False
        self.optimizer_skips_consecutive = 0
        interval = int(self.capability_config["validation_interval"])
        final_step = int(self.capability_config["scheduler"]["total_steps"])
        event_due = self.global_step % interval == 0 or self.global_step == final_step
        if event_due:
            event = self._run_validation_event()
            self._apply_abort_guards(event)
            self._save_best_checkpoints(event)
            # Persist the completed event and updated domain-best state before
            # another optimizer update can begin.
            self.save_training_checkpoint(
                Path(self.config.checkpoint_dir) / "latest.pt",
                loss=float(event["fineweb_loss"]),
            )
        checkpoint_interval = int(self.capability_config["checkpoint_interval"])
        if self.global_step % checkpoint_interval == 0:
            periodic = (
                Path(self.config.checkpoint_dir) / f"step_{self.global_step}.pt"
            )
            self.save_training_checkpoint(periodic)
            removed = apply_checkpoint_retention(
                Path(self.config.checkpoint_dir),
                keep_periodic=int(
                    self.capability_config["checkpoint_retention"]["periodic_keep"]
                ),
                milestones=self.capability_config["checkpoint_retention"][
                    "milestone_steps"
                ],
            )
            for path in removed:
                print(f"checkpoint_retention_removed={path}")
        thermal_policy = self.capability_config["thermal_safety"]
        if self.global_step % int(thermal_policy["monitor_interval_steps"]) == 0:
            reading = self.thermal_monitor.read(self.global_step)
            if not reading["available"] and bool(
                thermal_policy["required_for_authorized_run"]
            ):
                self.abort_reason = "temperature_monitor_unavailable"
                raise RuntimeError("GPU temperature monitoring is unavailable")
            if reading.get("warning"):
                print(f"thermal_warning={reading['celsius']}C")
            if reading.get("abort"):
                self.abort_reason = "thermal_threshold"
                self.save_training_checkpoint(
                    Path(self.config.checkpoint_dir)
                    / f"thermal_stop_step_{self.global_step}.pt"
                )
                self.save_training_checkpoint(
                    Path(self.config.checkpoint_dir) / "latest.pt"
                )
                raise RuntimeError("sustained GPU thermal threshold reached")
        return True

    def fit(self) -> None:
        """Run exactly one scheduled epoch without the legacy mixed best metric."""

        if self.start_epoch >= self.config.epochs:
            print("Capability training already completed.")
            return
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        if self._resume_phase == "post_train_pre_validation":
            if not self.train_sampler.epoch_complete:
                raise RuntimeError("completed train phase has a non-exhausted sampler")
        else:
            self.train_epoch()
        if not self.train_sampler.epoch_complete:
            raise RuntimeError("capability epoch ended before schedule exhaustion")
        if not any(
            int(event["experiment_step"]) == self.global_step
            for event in self.validation_events
        ):
            event = self._run_validation_event()
            self._apply_abort_guards(event)
            self._save_best_checkpoints(event)
        self.train_sampler.advance_epoch()
        self.start_epoch = self.train_sampler.position.epoch
        self._resume_phase = "next_epoch"
        self._optimizer_steps_in_epoch = 0
        final_path = Path(self.config.checkpoint_dir) / "final.pt"
        self.save_training_checkpoint(final_path)
        self.save_training_checkpoint(Path(self.config.checkpoint_dir) / "latest.pt")
