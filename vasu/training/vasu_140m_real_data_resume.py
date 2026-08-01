"""Fail-closed contracts for VASU-140M real-data exact-resume evidence.

This module deliberately contains no training loop.  It validates the immutable
execution specification, binds input bytes before a future load, records exact
state digests, and validates the resulting qualification report.  Execution is
separately review- and authorization-gated.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch


SPECIFICATION_SCHEMA_ID = "vasu_140m_real_data_exact_resume_specification_v1"
RESULT_SCHEMA_ID = "vasu_140m_real_data_exact_resume_result_v1"
FAMILY_ID = "vasu_140m_v1"
RECORD_WIDTH = 513
SEQUENCE_LENGTH = 512
PROTECTED_PREFIXES = (
    "checkpoints/",
    "data/processed/",
    "data/manifests/",
)
STATE_COMPONENTS = frozenset(
    {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "gradients",
        "rng_python",
        "rng_numpy",
        "rng_torch_cpu",
        "rng_torch_cuda",
        "sampler",
        "validation",
    }
)
ADVERSARIAL_CHECKS = frozenset(
    {
        "truncated_checkpoint",
        "corrupted_checkpoint",
        "missing_sidecar",
        "corrupted_sidecar",
        "family_mismatch",
        "tokenizer_mismatch",
        "release_mismatch",
        "schedule_mismatch",
        "config_mismatch",
        "swapped_token_file",
        "swapped_mask_file",
        "changed_source_order",
        "changed_accumulation",
        "missing_partial_gradients",
        "invalid_scaler_state",
        "disk_exhaustion",
        "atomic_rename_failure",
        "stale_temporary_directory",
        "windows_open_handle",
        "post_validation_mutation",
    }
)


def canonical_json(value: object) -> bytes:
    """Return the one canonical JSON encoding used by qualification identities."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase 40-character Git commit")
    return text


def _positive_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _relative_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be repository-relative")
    return text


def _binding(value: object, label: str) -> tuple[str, str, int | None]:
    binding = _mapping(value, label)
    permitted = {"path", "sha256", "bytes"}
    if set(binding) not in ({"path", "sha256"}, permitted):
        _exact(binding, permitted, label)
    path = _relative_path(binding["path"], f"{label}.path")
    digest = _sha(binding["sha256"], f"{label}.sha256")
    byte_count = None
    if "bytes" in binding:
        byte_count = _positive_int(binding["bytes"], f"{label}.bytes")
    return path, digest, byte_count


def specification_sha256(specification: Mapping[str, object]) -> str:
    body = dict(specification)
    body.pop("specification_sha256", None)
    return sha256_bytes(canonical_json(body))


def validate_execution_specification(specification: Mapping[str, object]) -> None:
    """Validate the future one-shot workload without making it executable."""

    _exact(
        specification,
        {
            "schema_id",
            "qualification_id",
            "repository_commit",
            "family",
            "tokenizer",
            "release",
            "schedule",
            "evaluation_development",
            "workload",
            "optimizer",
            "scheduler",
            "runtime",
            "output",
            "review",
            "execution_authorized",
            "training_authorized",
            "specification_sha256",
        },
        "execution specification",
    )
    if specification["schema_id"] != SPECIFICATION_SCHEMA_ID:
        raise ValueError("execution specification schema mismatch")
    _string(specification["qualification_id"], "qualification_id")
    _commit(specification["repository_commit"], "repository_commit")

    family = _mapping(specification["family"], "family")
    _exact(family, {"family_id", "family_sha256", "config_sha256"}, "family")
    if family["family_id"] != FAMILY_ID:
        raise ValueError("real-data resume is restricted to vasu_140m_v1")
    _sha(family["family_sha256"], "family.family_sha256")
    _sha(family["config_sha256"], "family.config_sha256")
    _binding(specification["tokenizer"], "tokenizer")

    release = _mapping(specification["release"], "release")
    _exact(release, {"release_id", "manifest", "sources"}, "release")
    _string(release["release_id"], "release.release_id")
    _binding(release["manifest"], "release.manifest")
    sources = release["sources"]
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("release must bind at least two sources")
    source_ids: set[str] = set()
    for index, raw_source in enumerate(sources):
        label = f"release source {index}"
        source = _mapping(raw_source, label)
        _exact(
            source,
            {"source_id", "tokens", "mask", "lineage", "record_count"},
            label,
        )
        source_id = _string(source["source_id"], f"{label}.source_id")
        if source_id in source_ids:
            raise ValueError("release source IDs must be unique")
        source_ids.add(source_id)
        for field in ("tokens", "mask", "lineage"):
            _binding(source[field], f"{label}.{field}")
        _positive_int(source["record_count"], f"{label}.record_count")

    schedule = _mapping(specification["schedule"], "schedule")
    _exact(schedule, {"artifact", "source_order", "no_replacement"}, "schedule")
    _binding(schedule["artifact"], "schedule.artifact")
    if schedule["source_order"] != [source["source_id"] for source in sources]:
        raise ValueError("schedule source order must exactly match release order")
    if schedule["no_replacement"] is not True:
        raise ValueError("qualification schedule must be no-replacement")
    _binding(specification["evaluation_development"], "evaluation_development")

    workload = _mapping(specification["workload"], "workload")
    _exact(
        workload,
        {
            "record_width",
            "sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
            "optimizer_updates",
            "record_ids",
            "source_transitions",
            "interruptions",
        },
        "workload",
    )
    if workload["record_width"] != RECORD_WIDTH or workload["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("qualification requires 513-token records and 512 positions")
    if workload["batch_size"] != 1:
        raise ValueError("qualification batch size must be one")
    accumulation = _positive_int(
        workload["gradient_accumulation_steps"],
        "workload.gradient_accumulation_steps",
    )
    if accumulation < 2 or workload["optimizer_updates"] != 2:
        raise ValueError("qualification requires accumulation >=2 and exactly 2 updates")
    record_ids = workload["record_ids"]
    if not isinstance(record_ids, list) or len(record_ids) < accumulation * 2:
        raise ValueError("workload record IDs do not cover two updates")
    if any(not isinstance(item, str) or not item for item in record_ids):
        raise ValueError("workload record IDs must be non-empty strings")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("workload record IDs must be unique")
    transitions = workload["source_transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("workload must cross a source boundary")
    for transition in transitions:
        item = _mapping(transition, "source transition")
        _exact(item, {"after_record_id", "from_source", "to_source"}, "source transition")
        if item["after_record_id"] not in record_ids:
            raise ValueError("source transition references an unknown record")
        if item["from_source"] not in source_ids or item["to_source"] not in source_ids:
            raise ValueError("source transition references an unknown source")
        if item["from_source"] == item["to_source"]:
            raise ValueError("source transition must change sources")
    interruptions = workload["interruptions"]
    if not isinstance(interruptions, list) or len(interruptions) != 2:
        raise ValueError("workload requires exactly two interruptions")
    if {item.get("kind") for item in interruptions if isinstance(item, Mapping)} != {
        "partial_accumulation",
        "source_boundary",
    }:
        raise ValueError("partial-accumulation and source-boundary interruptions are required")
    for raw_interruption in interruptions:
        interruption = _mapping(raw_interruption, "interruption")
        _exact(interruption, {"kind", "after_record_id", "microbatches_in_accumulation"}, "interruption")
        if interruption["after_record_id"] not in record_ids:
            raise ValueError("interruption references an unknown record")
        microbatches = _positive_int(
            interruption["microbatches_in_accumulation"],
            "interruption.microbatches_in_accumulation",
            allow_zero=True,
        )
        if interruption["kind"] == "partial_accumulation" and not 0 < microbatches < accumulation:
            raise ValueError("partial interruption must preserve nonzero partial gradients")

    for label in ("optimizer", "scheduler"):
        value = _mapping(specification[label], label)
        _exact(value, {"implementation", "parameters_sha256"}, label)
        _binding(value["implementation"], f"{label}.implementation")
        _sha(value["parameters_sha256"], f"{label}.parameters_sha256")
    runtime = _mapping(specification["runtime"], "runtime")
    _exact(runtime, {"device", "device_index", "amp_dtype", "deterministic_algorithms"}, "runtime")
    if runtime["device"] != "cuda" or runtime["amp_dtype"] not in {"bfloat16", "float16"}:
        raise ValueError("runtime must bind CUDA and BF16 or FP16")
    _positive_int(runtime["device_index"], "runtime.device_index", allow_zero=True)
    if runtime["deterministic_algorithms"] is not True:
        raise ValueError("deterministic algorithms must be enabled")

    output = _mapping(specification["output"], "output")
    _exact(output, {"system_temporary_only", "checkpoint_relative_path", "result_path"}, "output")
    if output["system_temporary_only"] is not True:
        raise ValueError("qualification checkpoints must be system-temporary")
    checkpoint_path = _relative_path(output["checkpoint_relative_path"], "output.checkpoint_relative_path")
    if checkpoint_path.casefold().startswith(PROTECTED_PREFIXES):
        raise ValueError("qualification checkpoint targets a protected path")
    _relative_path(output["result_path"], "output.result_path")

    review = _mapping(specification["review"], "review")
    _exact(review, {"implementation_decision", "execution_decision"}, "review")
    _binding(review["implementation_decision"], "review.implementation_decision")
    if review["execution_decision"] is not None:
        _binding(review["execution_decision"], "review.execution_decision")
    if specification["execution_authorized"] is not False or specification["training_authorized"] is not False:
        raise ValueError("reviewable specification must remain non-authorizing")
    if _sha(specification["specification_sha256"], "specification_sha256") != specification_sha256(specification):
        raise ValueError("execution specification identity mismatch")


@dataclass(frozen=True)
class OpenBinding:
    """An open, byte-identity-checked artifact protected from path replacement."""

    path: Path
    handle: BinaryIO
    sha256: str
    byte_count: int

    def verify_unchanged(self) -> None:
        """Reject mutation after initial validation and reset for loading."""

        self.handle.seek(0)
        digest = hashlib.sha256()
        observed_bytes = 0
        for chunk in iter(lambda: self.handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            observed_bytes += len(chunk)
        if digest.hexdigest() != self.sha256 or observed_bytes != self.byte_count:
            raise ValueError(f"open artifact mutated after validation: {self.path}")
        self.handle.seek(0)

    def close(self) -> None:
        self.handle.close()


def open_verified_binding(
    repository_root: Path,
    binding: Mapping[str, object],
    *,
    label: str,
) -> OpenBinding:
    """Open and hash an artifact once, retaining the handle through loading."""

    relative, expected_sha, expected_bytes = _binding(binding, label)
    root = repository_root.resolve()
    unresolved = root / relative
    current = unresolved
    while current != root:
        if current.exists() and (current.is_symlink() or _is_junction(current)):
            raise ValueError(f"{label} may not traverse a link or junction")
        current = current.parent
    candidate = unresolved.resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"{label} is missing or escapes the repository")
    handle = candidate.open("rb")
    try:
        digest = hashlib.sha256()
        observed_bytes = 0
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            observed_bytes += len(chunk)
        observed_sha = digest.hexdigest()
        if observed_sha != expected_sha or (
            expected_bytes is not None and observed_bytes != expected_bytes
        ):
            raise ValueError(f"{label} byte identity mismatch")
        handle.seek(0)
        return OpenBinding(candidate, handle, observed_sha, observed_bytes)
    except Exception:
        handle.close()
        raise


def _is_junction(path: Path) -> bool:
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _digest_update(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"torch")
        digest.update(str(tensor.dtype).encode())
        digest.update(canonical_json(list(tensor.shape)))
        digest.update(memoryview(tensor.numpy()))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"numpy")
        digest.update(str(array.dtype).encode())
        digest.update(canonical_json(list(array.shape)))
        digest.update(memoryview(array))
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_update(digest, key)
            _digest_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _digest_update(digest, item)
    elif value is None or isinstance(value, (str, int, float, bool, bytes)):
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"unsupported state digest value: {type(value)!r}")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def build_state_evidence(
    components: Mapping[str, Any],
    *,
    consumed_record_ids: Sequence[str],
    source_transitions: Sequence[Mapping[str, object]],
    supervised_target_count: int,
    microbatch_count: int,
    optimizer_update_count: int,
) -> dict[str, object]:
    """Reduce runtime state to exact identities suitable for comparison."""

    if set(components) != STATE_COMPONENTS:
        raise ValueError("state evidence must contain every required component")
    records = list(consumed_record_ids)
    if not records or len(records) != len(set(records)):
        raise ValueError("consumed record IDs must be non-empty and unique")
    _positive_int(supervised_target_count, "supervised_target_count")
    _positive_int(microbatch_count, "microbatch_count")
    _positive_int(optimizer_update_count, "optimizer_update_count")
    evidence: dict[str, object] = {
        "component_sha256": {
            name: state_sha256(components[name]) for name in sorted(components)
        },
        "consumed_record_ids": records,
        "source_transitions": [dict(item) for item in source_transitions],
        "supervised_target_count": supervised_target_count,
        "microbatch_count": microbatch_count,
        "optimizer_update_count": optimizer_update_count,
    }
    evidence["state_sha256"] = sha256_bytes(canonical_json(evidence))
    return evidence


def validate_state_evidence(evidence: Mapping[str, object], label: str) -> None:
    """Validate the internal identity of one control or resumed state record."""

    fields = {
        "component_sha256",
        "consumed_record_ids",
        "source_transitions",
        "supervised_target_count",
        "microbatch_count",
        "optimizer_update_count",
        "state_sha256",
    }
    _exact(evidence, fields, label)
    components = _mapping(evidence["component_sha256"], f"{label}.component_sha256")
    if set(components) != STATE_COMPONENTS:
        raise ValueError(f"{label} component identities are incomplete")
    for name, digest in components.items():
        _sha(digest, f"{label}.component_sha256.{name}")
    records = evidence["consumed_record_ids"]
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(item, str) or not item for item in records)
        or len(records) != len(set(records))
    ):
        raise ValueError(f"{label} consumed record IDs are invalid")
    transitions = evidence["source_transitions"]
    if not isinstance(transitions, list) or not transitions:
        raise ValueError(f"{label} source transitions are missing")
    for raw in transitions:
        transition = _mapping(raw, f"{label} source transition")
        _exact(
            transition,
            {"after_record_id", "from_source", "to_source"},
            f"{label} source transition",
        )
        if transition["after_record_id"] not in records:
            raise ValueError(f"{label} source transition references an unknown record")
        if transition["from_source"] == transition["to_source"]:
            raise ValueError(f"{label} source transition does not change source")
    for field in (
        "supervised_target_count",
        "microbatch_count",
        "optimizer_update_count",
    ):
        _positive_int(evidence[field], f"{label}.{field}")
    reported = _sha(evidence["state_sha256"], f"{label}.state_sha256")
    body = dict(evidence)
    del body["state_sha256"]
    if sha256_bytes(canonical_json(body)) != reported:
        raise ValueError(f"{label} internal identity mismatch")


def compare_state_evidence(
    control: Mapping[str, object], resumed: Mapping[str, object]
) -> dict[str, bool]:
    """Compare exact state evidence; no numerical tolerance is accepted."""

    validate_state_evidence(control, "control state evidence")
    validate_state_evidence(resumed, "resumed state evidence")
    checks = {
        "components_exact": control["component_sha256"] == resumed["component_sha256"],
        "record_order_exact": control["consumed_record_ids"] == resumed["consumed_record_ids"],
        "source_transitions_exact": control["source_transitions"] == resumed["source_transitions"],
        "target_count_exact": control["supervised_target_count"] == resumed["supervised_target_count"],
        "progress_exact": (
            control["microbatch_count"] == resumed["microbatch_count"]
            and control["optimizer_update_count"] == resumed["optimizer_update_count"]
        ),
        "state_identity_exact": control["state_sha256"] == resumed["state_sha256"],
    }
    return checks


def result_sha256(report: Mapping[str, object]) -> str:
    body = dict(report)
    body.pop("result_sha256", None)
    return sha256_bytes(canonical_json(body))


def validate_result_report(report: Mapping[str, object]) -> None:
    """Validate completed evidence and keep it explicitly non-authorizing."""

    _exact(
        report,
        {
            "schema_id",
            "qualification_id",
            "specification_sha256",
            "repository_commit",
            "implementation_sha256",
            "environment",
            "control",
            "resumed",
            "comparison_checks",
            "checkpoint_evidence",
            "adversarial_checks",
            "cleanup_complete",
            "passed",
            "execution_authorized",
            "training_authorized",
            "result_sha256",
        },
        "result report",
    )
    if report["schema_id"] != RESULT_SCHEMA_ID:
        raise ValueError("result report schema mismatch")
    _string(report["qualification_id"], "qualification_id")
    _sha(report["specification_sha256"], "specification_sha256")
    _commit(report["repository_commit"], "repository_commit")
    _sha(report["implementation_sha256"], "implementation_sha256")
    _mapping(report["environment"], "environment")
    comparison = _mapping(report["comparison_checks"], "comparison_checks")
    required_comparison = {
        "components_exact",
        "record_order_exact",
        "source_transitions_exact",
        "target_count_exact",
        "progress_exact",
        "state_identity_exact",
    }
    _exact(comparison, required_comparison, "comparison_checks")
    if not all(value is True for value in comparison.values()):
        raise ValueError("result contains a failed exact-state comparison")
    checkpoint = _mapping(report["checkpoint_evidence"], "checkpoint_evidence")
    _exact(
        checkpoint,
        {"sha256", "bytes", "sidecar_sha256", "atomic_write", "strict_reload"},
        "checkpoint_evidence",
    )
    _sha(checkpoint["sha256"], "checkpoint_evidence.sha256")
    _sha(checkpoint["sidecar_sha256"], "checkpoint_evidence.sidecar_sha256")
    _positive_int(checkpoint["bytes"], "checkpoint_evidence.bytes")
    if checkpoint["atomic_write"] is not True or checkpoint["strict_reload"] is not True:
        raise ValueError("checkpoint integrity evidence is incomplete")
    adversarial = _mapping(report["adversarial_checks"], "adversarial_checks")
    if set(adversarial) != ADVERSARIAL_CHECKS or not all(
        value is True for value in adversarial.values()
    ):
        raise ValueError("adversarial qualification is incomplete")
    expected_comparison = compare_state_evidence(
        _mapping(report["control"], "control"),
        _mapping(report["resumed"], "resumed"),
    )
    if dict(comparison) != expected_comparison:
        raise ValueError("reported comparison does not match state evidence")
    if report["cleanup_complete"] is not True or report["passed"] is not True:
        raise ValueError("qualification did not pass cleanly")
    if report["execution_authorized"] is not False or report["training_authorized"] is not False:
        raise ValueError("qualification result may not authorize execution or training")
    if _sha(report["result_sha256"], "result_sha256") != result_sha256(report):
        raise ValueError("qualification result identity mismatch")
