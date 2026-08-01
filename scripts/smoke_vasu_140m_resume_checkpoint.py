"""Frozen, non-training smoke for VASU-140M qualification checkpoints."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.training.vasu_140m_real_data_resume import (  # noqa: E402
    STATE_COMPONENTS,
    canonical_json,
    state_sha256,
)
from vasu.training.vasu_140m_resume_checkpoint import (  # noqa: E402
    build_checkpoint_payload,
    load_verified_qualification_checkpoint,
    sidecar_sha256,
    write_qualification_checkpoint,
)


PARENT_COMMIT = "eafd4d708415116663a5c1b1909cc94a6e7b00b3"
SPECIFICATION_SHA256 = "1" * 64
IDENTITIES = {
    "evaluation_development": "2" * 64,
    "release": "3" * 64,
    "schedule": "4" * 64,
    "tokenizer": "5" * 64,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_state() -> dict[str, object]:
    state: dict[str, object] = {}
    for index, name in enumerate(sorted(STATE_COMPONENTS)):
        state[name] = {
            "ordinal": index,
            "tensor": torch.tensor([index, index + 1], dtype=torch.int64),
        }
    return state


def build_report() -> dict[str, object]:
    state = _fixture_state()
    payload = build_checkpoint_payload(
        qualification_id="vasu-140m-resume-checkpoint-fixture-v1",
        specification_sha256=SPECIFICATION_SHA256,
        repository_commit=PARENT_COMMIT,
        creation_phase="partial_accumulation",
        identities=IDENTITIES,
        progress={
            "consumed_record_ids": ["source-a-record-0001"],
            "source_index": 0,
            "record_index": 1,
            "microbatch_count": 1,
            "optimizer_update_count": 0,
            "accumulated_microbatches": 1,
            "supervised_target_count": 511,
        },
        state=state,
    )
    with tempfile.TemporaryDirectory(
        prefix="vasu_140m_resume_checkpoint_smoke_"
    ) as raw_root:
        root = Path(raw_root)
        sidecar = write_qualification_checkpoint(root, "interruption.pt", payload)
        restored, observed_sidecar = load_verified_qualification_checkpoint(
            root,
            "interruption.pt",
            expected_specification_sha256=SPECIFICATION_SHA256,
            expected_identities=IDENTITIES,
        )
        checks = {
            "payload_exact": state_sha256(restored) == state_sha256(payload),
            "sidecar_exact": observed_sidecar == sidecar,
            "sidecar_identity_exact": (
                sidecar_sha256(observed_sidecar) == observed_sidecar["sidecar_sha256"]
            ),
            "state_identity_exact": restored["state_sha256"] == state_sha256(state),
            "system_temporary_only": root.is_relative_to(
                Path(tempfile.gettempdir()).resolve()
            ),
        }
        checkpoint_evidence = {
            "checkpoint_sha256": sidecar["checkpoint_sha256"],
            "checkpoint_bytes": sidecar["checkpoint_bytes"],
            "sidecar_sha256": sidecar["sidecar_sha256"],
            "state_sha256": sidecar["state_sha256"],
        }
    checks["cleanup_complete"] = not root.exists()
    report: dict[str, object] = {
        "schema_id": "vasu_140m_resume_checkpoint_qualification_v1",
        "repository_parent": PARENT_COMMIT,
        "implementation_sha256": _sha256_file(
            REPOSITORY_ROOT / "vasu/training/vasu_140m_resume_checkpoint.py"
        ),
        "test_sha256": _sha256_file(
            REPOSITORY_ROOT / "tests/test_vasu_140m_resume_checkpoint.py"
        ),
        "specification_sha256": SPECIFICATION_SHA256,
        "bound_identities": IDENTITIES,
        "checkpoint_evidence": checkpoint_evidence,
        "checks": checks,
        "passed": all(checks.values()),
        "model_created": False,
        "optimizer_created": False,
        "cuda_invoked": False,
        "optimizer_update_performed": False,
        "training_authorized": False,
    }
    report["qualification_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def main() -> int:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
