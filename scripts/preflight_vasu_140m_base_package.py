"""Run the non-authorizing VASU-140M base-package final preflight."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qualify_vasu_140m_cuda_amp import read_telemetry  # noqa: E402
from vasu.training.vasu_140m_final_preflight import (  # noqa: E402
    FinalPreflightObservation,
    build_final_preflight_report,
    sha256_file,
)
from vasu.training.vasu_140m_real_data_resume import (  # noqa: E402
    validate_result_report as validate_resume_result,
)


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.is_dir():
        if candidate.parent == candidate:
            raise ValueError("no existing ancestor for atomic-replace probe")
        candidate = candidate.parent
    return candidate


def _probe_atomic_replace(checkpoint_directory: Path) -> bool:
    parent = _nearest_existing_directory(checkpoint_directory)
    with tempfile.TemporaryDirectory(
        prefix=".vasu-140m-final-preflight-",
        dir=parent,
    ) as temporary:
        root = Path(temporary)
        source = root / "source.tmp"
        destination = root / "destination.tmp"
        source.write_bytes(b"replacement")
        destination.write_bytes(b"original")
        os.replace(source, destination)
        return not source.exists() and destination.read_bytes() == b"replacement"


def _resume_capabilities(plan: dict[str, object]) -> tuple[bool, bool]:
    binding = plan["gate_evidence"]["real_data_exact_resume"]["artifact"]
    path = (REPOSITORY_ROOT / binding["path"]).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT) or not path.is_file():
        raise ValueError("real-data exact-resume result is missing")
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("real-data exact-resume result identity mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_resume_result(result)
    checkpoint = result["checkpoint_evidence"]
    sidecars = (
        isinstance(checkpoint.get("sidecar_sha256"), str)
        and len(checkpoint["sidecar_sha256"]) == 64
        and checkpoint.get("atomic_write") is True
        and checkpoint.get("strict_reload") is True
    )
    comparison = result["comparison_checks"]
    exact_resume = bool(comparison) and all(comparison.values())
    return sidecars, exact_resume


def _observation(plan: dict[str, object], device_index: int) -> FinalPreflightObservation:
    cuda_available = torch.cuda.is_available() and 0 <= device_index < torch.cuda.device_count()
    if cuda_available:
        amp_dtype = "bf16" if torch.cuda.is_bf16_supported(device_index) else "fp16"
        telemetry = read_telemetry(device_index)
    else:
        amp_dtype = "unavailable"
        telemetry = {"available": False, "provider": "nvidia-smi"}
    checkpoint_directory = REPOSITORY_ROOT / plan["outputs"]["checkpoint_directory"]
    sidecars, exact_resume = _resume_capabilities(plan)
    return FinalPreflightObservation(
        worktree_clean=_git("status", "--porcelain=v1") == "",
        runtime_commit=_git("rev-parse", "HEAD"),
        cuda_available=cuda_available,
        cuda_device_index=device_index,
        amp_dtype=amp_dtype,
        free_disk_bytes=shutil.disk_usage(REPOSITORY_ROOT).free,
        telemetry_available=telemetry.get("available") is True,
        telemetry_provider=str(telemetry.get("provider", "unavailable")),
        temperature_celsius=(
            float(telemetry["temperature_celsius"])
            if telemetry.get("temperature_celsius") is not None
            else None
        ),
        atomic_replace_supported=_probe_atomic_replace(checkpoint_directory),
        checkpoint_sidecar_supported=sidecars,
        explicit_resume_enforced=exact_resume,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--plan-decision", required=True)
    parser.add_argument("--plan-decision-sha256", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = build_final_preflight_report(
        plan_path=plan_path,
        plan_sha256=args.plan_sha256,
        plan_decision_path=args.plan_decision,
        plan_decision_sha256=args.plan_decision_sha256,
        repository_root=REPOSITORY_ROOT,
        observation=_observation(plan, args.device),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["eligible_for_authorization_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
