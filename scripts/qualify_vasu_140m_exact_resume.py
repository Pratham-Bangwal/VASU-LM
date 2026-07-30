"""Synthetic exact-resume qualification for VASU-140M-v1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.qualify_vasu_140m_cpu import (  # noqa: E402
    report_sha256,
    write_immutable_report,
)
from vasu.config import get_vasu_140m_config  # noqa: E402
from vasu.model import VASUModel, validate_checkpoint_family_identity  # noqa: E402
from vasu.training.trainer import Trainer  # noqa: E402

SEED = 140_044
MINIMUM_FREE_DISK_BYTES = 4 * 1024**3


class SyntheticResumeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Four immutable records with externally visible consumption order."""

    def __init__(self, seen: list[int]) -> None:
        self.seen = seen
        generator = torch.Generator().manual_seed(90210)
        tokens = torch.randint(0, 32_000, (4, 5), generator=generator)
        self.inputs = tokens[:, :-1]
        self.targets = tokens[:, 1:]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen.append(index)
        return self.inputs[index], self.targets[index]

    def resume_identity(self) -> dict[str, str]:
        return {
            "schema": "vasu.synthetic-resume-dataset.v1",
            "sha256": state_sha256(
                {"inputs": self.inputs, "targets": self.targets}
            ),
        }


def _seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _config(directory: Path, checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model_family_id="vasu_140m_v1",
        batch_size=1,
        gradient_accumulation_steps=2,
        grad_clip=1.0,
        epochs=1,
        learning_rate=1e-5,
        minimum_learning_rate=1e-6,
        weight_decay=0.01,
        optimizer_backend="standard",
        scheduler_total_steps=2,
        warmup_steps=0,
        use_amp=False,
        seed=SEED,
        shuffle=True,
        drop_last=True,
        save_every_steps=0,
        checkpoint_path=str(checkpoint),
        checkpoint_dir=str(directory / "checkpoints"),
        num_workers=0,
        persistent_workers=False,
    )


def _digest_update(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(memoryview(tensor.numpy()))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _digest_update(digest, key)
            _digest_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _digest_update(digest, item)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"unsupported digest value: {type(value)!r}")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def _final_summary(trainer: Trainer, seen: list[int]) -> dict[str, Any]:
    return {
        "model_sha256": state_sha256(trainer.model.state_dict()),
        "optimizer_sha256": state_sha256(trainer.optimizer.state_dict()),
        "scheduler_sha256": state_sha256(trainer.scheduler.state_dict()),
        "scaler_sha256": state_sha256(trainer.scaler.state_dict()),
        "sampler_state": trainer.train_sampler.state_dict(),
        "global_step": trainer.global_step,
        "accumulated_microbatches": trainer._accumulated_microbatches,
        "optimizer_steps_in_epoch": trainer._optimizer_steps_in_epoch,
        "resume_phase": trainer._resume_phase,
        "seen_indices": seen,
        "next_rng": {
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch": torch.rand(4).tolist(),
        },
    }


def _run_worker(mode: str, directory: Path) -> dict[str, Any]:
    _seed_everything()
    seen: list[int] = []
    checkpoint = directory / "interruption.pt"
    config = _config(directory, checkpoint)
    trainer = Trainer(
        VASUModel(get_vasu_140m_config()),
        tokenizer=None,
        train_dataset=SyntheticResumeDataset(seen),
        val_dataset=SyntheticResumeDataset([]),
        config=config,
        device=torch.device("cpu"),
    )

    interruption: dict[str, Any] | None = None
    if mode == "control":
        trainer.train_epoch()
    elif mode == "resumed":
        trainer.train_epoch(max_microbatches=1)
        trainer.save_training_checkpoint(checkpoint)
        checkpoint_sha256 = _sha256_file(checkpoint)
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        validate_checkpoint_family_identity(payload, "vasu_140m_v1")
        progress = payload["training_progress"]
        interruption = {
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "global_step": payload["global_step"],
            "next_batch_index": progress["sampler"]["next_batch_index"],
            "accumulated_microbatches": progress["accumulated_microbatches"],
            "gradient_tensor_count": len(progress["gradients"]),
            "dataset_identity": progress["dataset_identity"],
            "family_identity_present": "model_family_identity" in payload,
        }
        del payload, trainer
        gc.collect()
        trainer = Trainer(
            VASUModel(get_vasu_140m_config()),
            tokenizer=None,
            train_dataset=SyntheticResumeDataset(seen),
            val_dataset=SyntheticResumeDataset([]),
            config=config,
            device=torch.device("cpu"),
        )
        interruption["restored"] = {
            "exact_resume_available": trainer.exact_resume_available,
            "next_batch_index": (
                trainer.train_sampler.position.next_batch_index
            ),
            "accumulated_microbatches": trainer._accumulated_microbatches,
            "global_step": trainer.global_step,
        }
        trainer.train_epoch()
    else:
        raise ValueError(f"unsupported worker mode: {mode}")

    return {
        "mode": mode,
        "final": _final_summary(trainer, seen),
        "interruption": interruption,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _worker_command(mode: str, directory: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        mode,
        "--directory",
        str(directory),
        "--worker-output",
        str(output),
    ]


def _run_subprocess(mode: str, directory: Path) -> dict[str, Any]:
    output = directory / f"{mode}.json"
    completed = subprocess.run(
        _worker_command(mode, directory, output),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"{mode} worker failed ({completed.returncode}): "
            f"{completed.stderr[-2000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def build_qualification_report() -> dict[str, Any]:
    free_before = shutil.disk_usage(REPOSITORY_ROOT).free
    if free_before < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            "exact-resume qualification requires at least 4 GiB free disk"
        )
    root = REPOSITORY_ROOT / "tmp" / "qualification"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="vasu_140m_exact_resume_",
        dir=root,
    ) as temporary:
        directory = Path(temporary)
        control = _run_subprocess("control", directory)
        resumed = _run_subprocess("resumed", directory)
        expected = control["final"]
        actual = resumed["final"]
        interruption = resumed["interruption"]
        dataset_identity = SyntheticResumeDataset([]).resume_identity()
        checks = {
            "model_exact": expected["model_sha256"]
            == actual["model_sha256"],
            "optimizer_exact": expected["optimizer_sha256"]
            == actual["optimizer_sha256"],
            "scheduler_exact": expected["scheduler_sha256"]
            == actual["scheduler_sha256"],
            "scaler_exact": expected["scaler_sha256"]
            == actual["scaler_sha256"],
            "sampler_exact": expected["sampler_state"]
            == actual["sampler_state"],
            "sample_order_exact": expected["seen_indices"]
            == actual["seen_indices"],
            "rng_exact": expected["next_rng"] == actual["next_rng"],
            "progress_exact": all(
                expected[name] == actual[name]
                for name in (
                    "global_step",
                    "accumulated_microbatches",
                    "optimizer_steps_in_epoch",
                    "resume_phase",
                )
            ),
            "partial_gradient_checkpoint": (
                interruption["global_step"] == 0
                and interruption["next_batch_index"] == 1
                and interruption["accumulated_microbatches"] == 1
                and interruption["gradient_tensor_count"] == 110
            ),
            "dataset_identity": interruption["dataset_identity"]
            == dataset_identity,
            "family_identity": interruption["family_identity_present"],
            "resume_position": interruption["restored"]
            == {
                "exact_resume_available": True,
                "next_batch_index": 1,
                "accumulated_microbatches": 1,
                "global_step": 0,
            },
        }
        checkpoint_observation = {
            key: value
            for key, value in interruption.items()
            if key not in {"dataset_identity", "restored"}
        }

    temporary_removed = not directory.exists()
    checks["temporary_artifacts_removed"] = temporary_removed
    return {
        "schema": "vasu.model-family-exact-resume-qualification.v1",
        "family_id": "vasu_140m_v1",
        "workload": {
            "seed": SEED,
            "synthetic_records": 4,
            "sequence_length": 4,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "optimizer": "AdamW-standard",
            "scheduler": "LambdaLR",
            "optimizer_updates": 2,
            "interruption_after_microbatches": 1,
            "real_data": False,
        },
        "checkpoint": checkpoint_observation,
        "control": expected,
        "resumed": actual,
        "checks": checks,
        "passed": all(checks.values()),
        "training_authorized": False,
        "free_disk_bytes_before": free_before,
        "free_disk_bytes_after": shutil.disk_usage(REPOSITORY_ROOT).free,
        "remaining_gates": [
            "cuda_memory_throughput_thermal",
            "513_token_data_release",
            "frozen_evaluation_baselines",
            "scientific_plan_and_hash_bound_authorization",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worker",
        choices=("control", "resumed"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker is not None:
        if args.directory is None or args.worker_output is None:
            raise ValueError("worker mode requires directory and output")
        report = _run_worker(args.worker, args.directory)
        args.worker_output.write_text(
            json.dumps(report, sort_keys=True),
            encoding="utf-8",
        )
        return 0

    report = build_qualification_report()
    report["report_sha256"] = report_sha256(report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if args.output is not None:
        write_immutable_report(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
