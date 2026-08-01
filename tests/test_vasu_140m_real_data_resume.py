from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from vasu.training.vasu_140m_real_data_resume import (
    ADVERSARIAL_CHECKS,
    FAMILY_ID,
    RESULT_SCHEMA_ID,
    SPECIFICATION_SCHEMA_ID,
    STATE_COMPONENTS,
    build_state_evidence,
    compare_state_evidence,
    open_verified_binding,
    result_sha256,
    specification_sha256,
    state_sha256,
    validate_execution_specification,
    validate_result_report,
    validate_state_evidence,
)


SHA = "a" * 64
COMMIT = "b" * 40


def binding(path: str, *, size: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": path, "sha256": SHA}
    if size is not None:
        value["bytes"] = size
    return value


def valid_specification() -> dict[str, object]:
    specification: dict[str, object] = {
        "schema_id": SPECIFICATION_SCHEMA_ID,
        "qualification_id": "vasu-140m-real-resume-fixture-v1",
        "repository_commit": COMMIT,
        "family": {
            "family_id": FAMILY_ID,
            "family_sha256": SHA,
            "config_sha256": SHA,
        },
        "tokenizer": binding("tokenizer/tokenizer.json"),
        "release": {
            "release_id": "vasu_140m_base_pretraining_v1",
            "manifest": binding("data/manifests/vasu_140m/base/v1.json"),
            "sources": [
                {
                    "source_id": "source_a",
                    "tokens": binding("data/source_a.tokens.bin"),
                    "mask": binding("data/source_a.mask.bin"),
                    "lineage": binding("data/source_a.lineage.jsonl"),
                    "record_count": 2,
                },
                {
                    "source_id": "source_b",
                    "tokens": binding("data/source_b.tokens.bin"),
                    "mask": binding("data/source_b.mask.bin"),
                    "lineage": binding("data/source_b.lineage.jsonl"),
                    "record_count": 2,
                },
            ],
        },
        "schedule": {
            "artifact": binding("data/schedules/base-v1.json"),
            "source_order": ["source_a", "source_b"],
            "no_replacement": True,
        },
        "evaluation_development": binding("evaluation/suites/base-v2-dev.json"),
        "workload": {
            "record_width": 513,
            "sequence_length": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "optimizer_updates": 2,
            "record_ids": ["a-0", "a-1", "b-0", "b-1"],
            "source_transitions": [
                {
                    "after_record_id": "a-1",
                    "from_source": "source_a",
                    "to_source": "source_b",
                }
            ],
            "interruptions": [
                {
                    "kind": "partial_accumulation",
                    "after_record_id": "a-0",
                    "microbatches_in_accumulation": 1,
                },
                {
                    "kind": "source_boundary",
                    "after_record_id": "a-1",
                    "microbatches_in_accumulation": 0,
                },
            ],
        },
        "optimizer": {
            "implementation": binding("vasu/training/optimizer.py"),
            "parameters_sha256": SHA,
        },
        "scheduler": {
            "implementation": binding("vasu/training/trainer.py"),
            "parameters_sha256": SHA,
        },
        "runtime": {
            "device": "cuda",
            "device_index": 0,
            "amp_dtype": "bfloat16",
            "deterministic_algorithms": True,
        },
        "output": {
            "system_temporary_only": True,
            "checkpoint_relative_path": "vasu_140m_resume/interruption.pt",
            "result_path": "evaluation/results/vasu_140m/resume.json",
        },
        "review": {
            "implementation_decision": binding("docs/implementation-decision.md"),
            "execution_decision": None,
        },
        "execution_authorized": False,
        "training_authorized": False,
    }
    specification["specification_sha256"] = specification_sha256(specification)
    return specification


def test_valid_execution_specification_is_non_authorizing() -> None:
    validate_execution_specification(valid_specification())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_id="wrong"), "schema"),
        (lambda value: value["family"].update(family_id="vasu_60m_v1"), "restricted"),
        (lambda value: value["release"].update(sources=value["release"]["sources"][:1]), "two sources"),
        (lambda value: value["schedule"].update(source_order=["source_b", "source_a"]), "source order"),
        (lambda value: value["workload"].update(record_width=257), "513-token"),
        (lambda value: value["workload"].update(optimizer_updates=3), "exactly 2"),
        (lambda value: value["runtime"].update(device="cpu"), "CUDA"),
        (lambda value: value["output"].update(checkpoint_relative_path="checkpoints/bad.pt"), "protected"),
        (lambda value: value.update(execution_authorized=True), "non-authorizing"),
    ],
)
def test_specification_drift_fails_closed(mutation, message: str) -> None:
    specification = valid_specification()
    mutation(specification)
    specification["specification_sha256"] = specification_sha256(specification)
    with pytest.raises(ValueError, match=message):
        validate_execution_specification(specification)


def test_specification_hash_rejects_unbound_mutation() -> None:
    specification = valid_specification()
    specification["runtime"]["amp_dtype"] = "float16"
    with pytest.raises(ValueError, match="identity"):
        validate_execution_specification(specification)


def test_open_verified_binding_detects_post_validation_mutation(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"immutable input")
    expected = {
        "path": "artifact.bin",
        "sha256": hashlib.sha256(b"immutable input").hexdigest(),
        "bytes": len(b"immutable input"),
    }
    opened = open_verified_binding(tmp_path, expected, label="fixture")
    try:
        artifact.write_bytes(b"mutated path")
        with pytest.raises(ValueError, match="mutated after validation"):
            opened.verify_unchanged()
    finally:
        opened.close()


def test_open_verified_binding_rejects_hash_and_size_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")
    with pytest.raises(ValueError, match="identity"):
        open_verified_binding(
            tmp_path,
            {"path": "artifact.bin", "sha256": SHA, "bytes": 7},
            label="fixture",
        )


def sample_components() -> dict[str, object]:
    return {
        name: {"name": name, "tensor": torch.tensor([1.0, 2.0])}
        for name in STATE_COMPONENTS
    }


def sample_evidence() -> dict[str, object]:
    return build_state_evidence(
        sample_components(),
        consumed_record_ids=["a-0", "a-1", "b-0", "b-1"],
        source_transitions=[
            {"after_record_id": "a-1", "from_source": "source_a", "to_source": "source_b"}
        ],
        supervised_target_count=2048,
        microbatch_count=4,
        optimizer_update_count=2,
    )


def test_state_digest_is_order_stable_and_tensor_sensitive() -> None:
    first = {"b": np.array([1, 2]), "a": torch.tensor([3.0])}
    second = {"a": torch.tensor([3.0]), "b": np.array([1, 2])}
    changed = {"a": torch.tensor([4.0]), "b": np.array([1, 2])}
    assert state_sha256(first) == state_sha256(second)
    assert state_sha256(first) != state_sha256(changed)


def test_exact_state_comparison_detects_component_and_order_drift() -> None:
    control = sample_evidence()
    resumed = copy.deepcopy(control)
    assert all(compare_state_evidence(control, resumed).values())
    resumed["consumed_record_ids"] = list(reversed(resumed["consumed_record_ids"]))
    with pytest.raises(ValueError, match="internal identity mismatch"):
        compare_state_evidence(control, resumed)


def test_state_evidence_rejects_missing_component_identity() -> None:
    evidence = sample_evidence()
    evidence["component_sha256"].pop("scaler")
    with pytest.raises(ValueError, match="incomplete"):
        validate_state_evidence(evidence, "fixture")


def valid_result() -> dict[str, object]:
    control = sample_evidence()
    resumed = copy.deepcopy(control)
    report: dict[str, object] = {
        "schema_id": RESULT_SCHEMA_ID,
        "qualification_id": "vasu-140m-real-resume-fixture-v1",
        "specification_sha256": SHA,
        "repository_commit": COMMIT,
        "implementation_sha256": SHA,
        "environment": {"torch": "fixture", "cuda": "fixture"},
        "control": control,
        "resumed": resumed,
        "comparison_checks": compare_state_evidence(control, resumed),
        "checkpoint_evidence": {
            "sha256": SHA,
            "bytes": 1024,
            "sidecar_sha256": SHA,
            "atomic_write": True,
            "strict_reload": True,
        },
        "adversarial_checks": {name: True for name in ADVERSARIAL_CHECKS},
        "cleanup_complete": True,
        "passed": True,
        "execution_authorized": False,
        "training_authorized": False,
    }
    report["result_sha256"] = result_sha256(report)
    return report


def test_valid_result_requires_complete_exact_and_adversarial_evidence() -> None:
    validate_result_report(valid_result())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["comparison_checks"].update(model_exact=False),
        lambda value: value["adversarial_checks"].pop("disk_exhaustion"),
        lambda value: value.update(cleanup_complete=False),
        lambda value: value.update(training_authorized=True),
    ],
)
def test_result_failures_cannot_report_acceptance(mutation) -> None:
    report = valid_result()
    mutation(report)
    report["result_sha256"] = result_sha256(report)
    with pytest.raises(ValueError):
        validate_result_report(report)
