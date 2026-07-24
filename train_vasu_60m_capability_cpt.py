"""Validate a capability-CPT launch config and enforce its authorization gate."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from vasu.config import get_vasu_60m_config
from evaluation.verified_arithmetic import load_verified_split, select_proxy_records
from vasu.model.model import VASUModel
from vasu.training.dataset import ManifestTokenDataset
from vasu.training.capability_cpt import (
    BLOCKED_MESSAGE,
    require_training_authorization,
    validate_capability_config,
)
from vasu.training.capability_runtime import (
    CapabilityTrainer,
    FlatTokenValidationDataset,
    ThermalMonitor,
    apply_checkpoint_retention,
    git_state,
    inspect_capability_checkpoint,
    require_disk_space,
)
from vasu.data.scheduled_mixture import sha256_file
from vasu.tokenizer.tokenizer import VASUTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate artifacts/configuration without checking authorization.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Explicit exact-resume checkpoint; implicit discovery is forbidden.",
    )
    parser.add_argument(
        "--retention-dry-run",
        action="store_true",
        help="List old periodic checkpoints that retention would remove.",
    )
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    result = validate_capability_config(args.config)
    config = result["config"]
    identity = result["capability_identity"]
    repository = git_state()
    summary = {
        "experiment_id": config["experiment_id"],
        "schedule_sha256": result["resolved_manifest"]["schedule"]["sha256"],
        "step_accounting": result["step_accounting"],
        "training_authorized": config["training_authorized"],
        "repository": repository,
        "capability_identity": identity,
    }
    print(json.dumps(summary, indent=2))
    checkpoint_dir = Path(config["checkpoint_directory"])
    if args.retention_dry_run:
        removed = apply_checkpoint_retention(
            checkpoint_dir,
            keep_periodic=int(config["checkpoint_retention"]["periodic_keep"]),
            milestones=config["checkpoint_retention"]["milestone_steps"],
            dry_run=True,
        ) if checkpoint_dir.is_dir() else []
        print(json.dumps({"retention_would_remove": [str(path) for path in removed]}, indent=2))
        return
    if args.validate_only:
        if not repository["clean"]:
            print("WARNING: validation ran with a dirty Git working tree.")
        return
    try:
        require_training_authorization(config)
    except PermissionError as error:
        raise SystemExit(BLOCKED_MESSAGE) from error
    if not repository["clean"]:
        raise SystemExit(
            "Authorized capability training requires a clean Git working tree."
        )
    if identity is None:
        raise SystemExit("Capability production identity is unavailable.")
    if args.resume_from is None:
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise SystemExit(
                "Output directory is not empty; use --resume-from explicitly."
            )
        checkpoint_path = checkpoint_dir / "__fresh_launch__.pt"
        resume_inspection = None
    else:
        if args.resume_from.name == "final.pt":
            raise SystemExit("final.pt is evaluation-only and cannot be resumed.")
        resume_inspection = inspect_capability_checkpoint(args.resume_from, identity)
        checkpoint_path = args.resume_from
        print(json.dumps({"resume": resume_inspection}, indent=2))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    require_disk_space(config, checkpoint_dir / "latest.pt")
    thermal = config["thermal_safety"]
    monitor = ThermalMonitor(
        warning_celsius=thermal["warning_celsius"],
        abort_celsius=thermal["abort_celsius"],
        critical_celsius=thermal["critical_celsius"],
        consecutive_abort_readings=thermal["consecutive_abort_readings"],
    )
    if not monitor.read(0)["available"]:
        raise SystemExit("Authorized launch requires GPU temperature monitoring.")

    parent = Path(config["parent_checkpoint"]["path"])
    if sha256_file(parent) != config["parent_checkpoint"]["sha256"]:
        raise ValueError("parent checkpoint SHA-256 mismatch")
    checkpoint = torch.load(
        parent,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    model = VASUModel(get_vasu_60m_config())
    model.load_state_dict(checkpoint["model"], strict=True)
    fineweb_spec = config["validation"]["fineweb"]
    fineweb_validation = ManifestTokenDataset(
        fineweb_spec["manifest"],
        "validation",
        config["sequence_length"],
        logical_start=0,
        logical_end=(
            int(fineweb_spec["maximum_records"]) * config["sequence_length"] + 1
        ),
    )
    wiki_spec = config["validation"]["wikimedia"]
    wikimedia_validation = FlatTokenValidationDataset(
        Path(wiki_spec["path"]),
        config["sequence_length"],
        expected_sha256=wiki_spec["sha256"],
        maximum_records=wiki_spec["maximum_records"],
    )
    arithmetic_spec = config["validation"]["arithmetic_proxy"]
    _, arithmetic_dev, _ = load_verified_split(
        Path(arithmetic_spec["manifest"]), "dev"
    )
    arithmetic_proxy = select_proxy_records(
        arithmetic_dev, int(arithmetic_spec["record_count"])
    )
    trainer_config = SimpleNamespace(
        batch_size=config["batch_size"],
        epochs=1,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        seed=config["seed"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        grad_clip=1.0,
        use_amp=True,
        optimizer_backend=config["optimizer_backend"],
        checkpoint_path=str(checkpoint_path),
        checkpoint_dir=config["checkpoint_directory"],
        save_every_steps=0,
        num_workers=0,
        persistent_workers=False,
        drop_last=True,
        shuffle=config["shuffle"],
        scheduler_total_steps=config["scheduler"]["total_steps"],
        warmup_steps=config["scheduler"]["warmup_steps"],
        minimum_learning_rate=config["scheduler"][
            "minimum_learning_rate"
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Authorized capability training requires CUDA.")
    tokenizer = VASUTokenizer()
    tokenizer.load(config["tokenizer"]["path"])
    loaders = {
        "fineweb": DataLoader(
            fineweb_validation,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        ),
        "wikimedia": DataLoader(
            wikimedia_validation,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
        ),
    }
    run_manifest_path = Path(config["log_directory"]) / "run_manifest.json"
    _atomic_json(
        run_manifest_path,
        {
            "experiment_id": config["experiment_id"],
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "repository_commit": repository["commit"],
            "capability_identity": identity,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "resume_inspection": resume_inspection,
            "abort_reason": None,
        },
    )
    trainer = CapabilityTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=result["dataset"],
        val_dataset=fineweb_validation,
        config=trainer_config,
        device=device,
        capability_config=config,
        capability_identity=identity,
        validation_loaders=loaders,
        arithmetic_records=arithmetic_proxy,
    )
    try:
        trainer.fit()
    except Exception as error:
        _atomic_json(
            run_manifest_path,
            {
                "experiment_id": config["experiment_id"],
                "status": "aborted",
                "ended_at": datetime.now(UTC).isoformat(),
                "repository_commit": repository["commit"],
                "capability_identity": identity,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "abort_reason": trainer.abort_reason or type(error).__name__,
                "error": str(error),
                "global_step": trainer.global_step,
            },
        )
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        raise
    _atomic_json(
        run_manifest_path,
        {
            "experiment_id": config["experiment_id"],
            "status": "complete",
            "ended_at": datetime.now(UTC).isoformat(),
            "repository_commit": repository["commit"],
            "capability_identity": identity,
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "abort_reason": None,
            "global_step": trainer.global_step,
        },
    )


if __name__ == "__main__":
    main()
