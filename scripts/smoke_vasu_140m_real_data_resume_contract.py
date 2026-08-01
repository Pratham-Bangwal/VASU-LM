"""Smoke the VASU-140M real-data exact-resume contracts without execution."""

from __future__ import annotations

import json
import sys
import copy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.training.vasu_140m_real_data_resume import (  # noqa: E402
    ADVERSARIAL_CHECKS,
    FAMILY_ID,
    RESULT_SCHEMA_ID,
    SPECIFICATION_SCHEMA_ID,
    STATE_COMPONENTS,
    build_state_evidence,
    compare_state_evidence,
    result_sha256,
    specification_sha256,
    validate_execution_specification,
    validate_result_report,
)

SHA = "a" * 64
COMMIT = "b" * 40


def _binding(path: str) -> dict[str, str]:
    return {"path": path, "sha256": SHA}


def _specification() -> dict[str, object]:
    sources = []
    for source_id in ("source_a", "source_b"):
        sources.append(
            {
                "source_id": source_id,
                "tokens": _binding(f"data/{source_id}.tokens.bin"),
                "mask": _binding(f"data/{source_id}.mask.bin"),
                "lineage": _binding(f"data/{source_id}.lineage.jsonl"),
                "record_count": 2,
            }
        )
    value: dict[str, object] = {
        "schema_id": SPECIFICATION_SCHEMA_ID,
        "qualification_id": "vasu-140m-real-resume-contract-fixture-v1",
        "repository_commit": COMMIT,
        "family": {"family_id": FAMILY_ID, "family_sha256": SHA, "config_sha256": SHA},
        "tokenizer": _binding("tokenizer/tokenizer.json"),
        "release": {
            "release_id": "fixture-release",
            "manifest": _binding("data/release.json"),
            "sources": sources,
        },
        "schedule": {
            "artifact": _binding("data/schedule.json"),
            "source_order": ["source_a", "source_b"],
            "no_replacement": True,
        },
        "evaluation_development": _binding("evaluation/dev.json"),
        "workload": {
            "record_width": 513,
            "sequence_length": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "optimizer_updates": 2,
            "record_ids": ["a-0", "a-1", "b-0", "b-1"],
            "source_transitions": [
                {"after_record_id": "a-1", "from_source": "source_a", "to_source": "source_b"}
            ],
            "interruptions": [
                {"kind": "partial_accumulation", "after_record_id": "a-0", "microbatches_in_accumulation": 1},
                {"kind": "source_boundary", "after_record_id": "a-1", "microbatches_in_accumulation": 0},
            ],
        },
        "optimizer": {"implementation": _binding("vasu/training/optimizer.py"), "parameters_sha256": SHA},
        "scheduler": {"implementation": _binding("vasu/training/trainer.py"), "parameters_sha256": SHA},
        "runtime": {"device": "cuda", "device_index": 0, "amp_dtype": "bfloat16", "deterministic_algorithms": True},
        "output": {
            "system_temporary_only": True,
            "checkpoint_relative_path": "vasu_140m_resume/interruption.pt",
            "result_path": "evaluation/results/vasu_140m/resume.json",
        },
        "review": {"implementation_decision": _binding("docs/implementation.md"), "execution_decision": None},
        "execution_authorized": False,
        "training_authorized": False,
    }
    value["specification_sha256"] = specification_sha256(value)
    return value


def _state_evidence() -> dict[str, object]:
    components = {name: {"fixture": name} for name in STATE_COMPONENTS}
    return build_state_evidence(
        components,
        consumed_record_ids=["a-0", "a-1", "b-0", "b-1"],
        source_transitions=[
            {"after_record_id": "a-1", "from_source": "source_a", "to_source": "source_b"}
        ],
        supervised_target_count=2048,
        microbatch_count=4,
        optimizer_update_count=2,
    )


def _result() -> dict[str, object]:
    control = _state_evidence()
    resumed = copy.deepcopy(control)
    value: dict[str, object] = {
        "schema_id": RESULT_SCHEMA_ID,
        "qualification_id": "vasu-140m-real-resume-contract-fixture-v1",
        "specification_sha256": SHA,
        "repository_commit": COMMIT,
        "implementation_sha256": SHA,
        "environment": {"scope": "contract-fixture"},
        "control": control,
        "resumed": resumed,
        "comparison_checks": compare_state_evidence(control, resumed),
        "checkpoint_evidence": {
            "sha256": SHA,
            "bytes": 1,
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
    value["result_sha256"] = result_sha256(value)
    return value


def build_report() -> dict[str, object]:
    specification = _specification()
    validate_execution_specification(specification)
    control = _state_evidence()
    result = _result()
    validate_result_report(result)
    return {
        "schema_id": "vasu_140m_real_data_resume_contract_qualification_v1",
        "specification_schema_valid": True,
        "specification_sha256": specification_sha256(specification),
        "result_schema_valid": True,
        "result_sha256": result_sha256(result),
        "exact_comparison_fields": sorted(compare_state_evidence(control, control)),
        "torch_cuda_invoked": False,
        "optimizer_created": False,
        "optimizer_update_performed": False,
        "checkpoint_created": False,
        "production_data_opened": False,
        "execution_authorized": False,
        "training_authorized": False,
    }


def main() -> int:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
