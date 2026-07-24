"""Validate a capability-CPT launch config and enforce its authorization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from vasu.config import get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.training.dataset import ManifestTokenDataset
from vasu.training.capability_cpt import (
    BLOCKED_MESSAGE,
    require_training_authorization,
    validate_capability_config,
)
from vasu.training.trainer import Trainer
from vasu.data.scheduled_mixture import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate artifacts/configuration without checking authorization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_capability_config(args.config)
    summary = {
        "experiment_id": result["config"]["experiment_id"],
        "schedule_sha256": result["resolved_manifest"]["schedule"]["sha256"],
        "step_accounting": result["step_accounting"],
        "training_authorized": result["config"]["training_authorized"],
    }
    print(json.dumps(summary, indent=2))
    if args.validate_only:
        return
    try:
        require_training_authorization(result["config"])
    except PermissionError as error:
        raise SystemExit(BLOCKED_MESSAGE) from error
    config = result["config"]
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
    validation = ManifestTokenDataset(
        "data/processed/pretrain/fineweb_manifest.json",
        "validation",
        config["sequence_length"],
        logical_start=0,
        logical_end=512 * config["sequence_length"] + 1,
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
        checkpoint_path=str(
            Path(config["checkpoint_directory"]) / "latest.pt"
        ),
        checkpoint_dir=config["checkpoint_directory"],
        save_every_steps=config["checkpoint_interval"],
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
    trainer = Trainer(
        model=model,
        tokenizer=None,
        train_dataset=result["dataset"],
        val_dataset=validation,
        config=trainer_config,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
