"""Disposable checkpoint round-trip qualification for VASU-140M-v1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qualify_vasu_140m_cpu import (  # noqa: E402
    report_sha256,
    write_immutable_report,
)
from vasu.config import get_vasu_140m_config, get_vasu_60m_config  # noqa: E402
from vasu.model import (  # noqa: E402
    VASUModel,
    build_model_family_identity,
    load_family_model_state,
    validate_checkpoint_family_identity,
    validate_family_config,
)
from vasu.training.checkpoint import save_checkpoint  # noqa: E402

MINIMUM_FREE_DISK_BYTES = 2 * 1024**3
SEED = 140_043


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_state_match(
    expected: VASUModel,
    actual: VASUModel,
) -> tuple[bool, int]:
    expected_state = expected.state_dict()
    actual_state = actual.state_dict()
    if tuple(expected_state) != tuple(actual_state):
        return False, 0
    compared = 0
    for name, expected_tensor in expected_state.items():
        if not torch.equal(expected_tensor, actual_state[name]):
            return False, compared
        compared += 1
    return True, compared


def _assert_wrong_family_rejection(checkpoint: dict[str, Any]) -> dict[str, bool]:
    checks = {
        "wrong_declared_family_rejected": False,
        "wrong_destination_config_rejected": False,
    }
    try:
        validate_checkpoint_family_identity(checkpoint, "vasu_60m_v1")
    except ValueError:
        checks["wrong_declared_family_rejected"] = True

    with torch.device("meta"):
        wrong_model = VASUModel(get_vasu_60m_config())
    try:
        load_family_model_state(
            wrong_model,
            checkpoint,
            "vasu_140m_v1",
        )
    except ValueError:
        checks["wrong_destination_config_rejected"] = True
    return checks


def build_qualification_report() -> dict[str, Any]:
    torch.manual_seed(SEED)
    config = get_vasu_140m_config()
    family = validate_family_config("vasu_140m_v1", config)
    free_before = shutil.disk_usage(REPOSITORY_ROOT).free
    if free_before < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            "checkpoint qualification requires at least 2 GiB free disk"
        )

    temporary_root = REPOSITORY_ROOT / "tmp" / "qualification"
    temporary_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path: Path | None = None
    result: dict[str, Any]
    with tempfile.TemporaryDirectory(
        prefix="vasu_140m_checkpoint_",
        dir=temporary_root,
    ) as directory:
        checkpoint_path = Path(directory) / "round_trip.pt"
        model = VASUModel(config).to(device="cpu", dtype=torch.float32)
        identity = build_model_family_identity(family.family_id, config)

        started = time.perf_counter()
        save_checkpoint(
            model=model,
            optimizer=None,
            scheduler=None,
            epoch=0,
            loss=None,
            path=checkpoint_path,
            global_step=0,
            verify_after_write=True,
            fsync=True,
            model_family_identity=identity,
            training_authorized=False,
            qualification_only=True,
        )
        save_seconds = time.perf_counter() - started
        checkpoint_bytes = checkpoint_path.stat().st_size
        checkpoint_sha256 = sha256_file(checkpoint_path)

        started = time.perf_counter()
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        load_seconds = time.perf_counter() - started
        validated = validate_checkpoint_family_identity(
            checkpoint,
            family.family_id,
        )
        wrong_family = _assert_wrong_family_rejection(checkpoint)

        restored = VASUModel(config).to(device="cpu", dtype=torch.float32)
        load_family_model_state(restored, checkpoint, family.family_id)
        tensors_equal, compared_tensors = exact_state_match(model, restored)
        checks = {
            "family_identity": validated is family,
            "strict_round_trip": tensors_equal,
            "weight_tying_restored": (
                restored.lm_head.weight
                is restored.embedding.embedding.weight
            ),
            "temporary_suffix_absent": not checkpoint_path.with_suffix(
                ".pt.tmp"
            ).exists(),
            **wrong_family,
        }
        result = {
            "schema": "vasu.model-family-checkpoint-qualification.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "family_id": family.family_id,
            "family_fingerprint": family.family_fingerprint,
            "config_fingerprint": family.config_fingerprint,
            "parameter_count": family.expected_parameter_count,
            "environment": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "device": "cpu",
                "dtype": "float32",
                "cuda_available": torch.cuda.is_available(),
            },
            "workload": {
                "seed": SEED,
                "model_only": True,
                "optimizer_created": False,
                "scheduler_created": False,
                "verify_after_write": True,
                "fsync": True,
                "temporary_checkpoint": True,
            },
            "checkpoint": {
                "bytes": checkpoint_bytes,
                "sha256": checkpoint_sha256,
                "save_seconds": save_seconds,
                "load_seconds": load_seconds,
                "state_tensor_count": compared_tensors,
                "payload_keys": sorted(checkpoint),
            },
            "checks": checks,
            "passed": all(checks.values()),
            "training_authorized": False,
            "remaining_gates": [
                "cuda_memory_throughput_thermal",
                "exact_resume",
                "513_token_data_release",
                "frozen_evaluation_baselines",
                "scientific_plan_and_hash_bound_authorization",
            ],
        }
        del checkpoint, restored, model
        gc.collect()

    assert checkpoint_path is not None
    result["temporary_checkpoint_removed"] = not checkpoint_path.exists()
    result["passed"] = (
        result["passed"] and result["temporary_checkpoint_removed"]
    )
    result["free_disk_bytes_before"] = free_before
    result["free_disk_bytes_after"] = shutil.disk_usage(REPOSITORY_ROOT).free
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional new immutable JSON evidence path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_qualification_report()
    report["report_sha256"] = report_sha256(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.output is not None:
        write_immutable_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
