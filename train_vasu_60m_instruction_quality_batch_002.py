"""Tiny masked instruction-quality refinement for VASU-60M.

Run --dry-run first. Training is isolated and requires explicit --train.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import train_vasu_60m_alpaca_masked_v2 as engine
from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.data.alpaca_masked_v2 import atomic_write_json
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer
from vasu.training.orchestration import inspect_checkpoint

EXPERIMENT_NAME = "vasu_60m_instruction_quality_batch_002_from_batch_001"
BASE_CHECKPOINT = Path(
    "checkpoints/vasu_60m/instruction_quality_batch_001_from_ultrachat_v2/best.pt"
)
BASE_GLOBAL_STEP = 201_304
BASE_CHECKPOINT_SHA256 = (
    "4f377fc49fef2382c531b5bb28731fa9664bac217256b8a76dcf1427f4d39d8e"
)
TOKENIZER_FILE = Path("assets/tokenizer.json")
DATA_FILE = Path(
    "data/processed/instruct/vasu_instruction_quality_v1_batch_002.bin"
)
MASK_FILE = Path(
    "data/processed/instruct/vasu_instruction_quality_v1_batch_002_mask.bin"
)
RELEASE_MANIFEST_FILE = Path(
    "data/manifests/instruct/vasu_instruction_quality_v1_batch_002_release.json"
)
CHECKPOINT_DIR = Path(
    "checkpoints/vasu_60m/instruction_quality_batch_002_from_batch_001"
)
MAIN_CHECKPOINT = CHECKPOINT_DIR / "vasu.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

EXPECTED_PARAMETERS = 58_337_792
EXPECTED_RELEASE_VERSION = "vasu_instruction_quality_release_v1"
EXPECTED_RECORDS = 95
TRAIN_RECORDS = 90
VALIDATION_RECORDS = 5
SEQUENCE_LENGTH = 256
RECORD_LENGTH = 257

EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-7
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_AMP = True
SEED = 42
SAVE_EVERY_STEPS = 1

PROTECTED_DIRECTORIES = (
    Path("checkpoints/vasu_60m/alpaca"),
    Path("checkpoints/vasu_60m/alpaca_masked_v2"),
    Path("checkpoints/vasu_60m/alpaca_masked_v3_from_200k"),
    Path("checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3"),
    Path("checkpoints/vasu_60m/instruction_quality_batch_001_from_ultrachat_v2"),
    Path("checkpoints/vasu_60m/factual_cpt_wikimedia_15pct_from_200k"),
    Path("checkpoints/vasu_60m/milestones"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_steps() -> int:
    micro_batches = TRAIN_RECORDS // BATCH_SIZE
    return math.ceil(micro_batches / GRADIENT_ACCUMULATION_STEPS)


def target_global_step() -> int:
    return BASE_GLOBAL_STEP + optimizer_steps()


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(RELEASE_MANIFEST_FILE.read_text(encoding="utf-8"))
    checks = {
        "release_version": EXPECTED_RELEASE_VERSION,
        "records": EXPECTED_RECORDS,
        "train_packed_records": TRAIN_RECORDS,
        "validation_packed_records": VALIDATION_RECORDS,
        "truncated_examples": 0,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"release manifest {key!r} must be {expected!r}; "
                f"found {manifest.get(key)!r}"
            )
    if manifest.get("cross_split_leakage") != []:
        raise ValueError("release reports cross-split leakage")
    return manifest


def validate_output_isolation() -> None:
    output = CHECKPOINT_DIR.resolve()
    for protected_path in PROTECTED_DIRECTORIES:
        protected = protected_path.resolve()
        if output == protected or protected in output.parents:
            raise ValueError(f"output overlaps protected path: {protected}")


def validate_base_checkpoint() -> None:
    result = inspect_checkpoint(
        BASE_CHECKPOINT, verify_finite=True, calculate_hash=True
    )
    if not result.valid:
        raise RuntimeError(f"invalid base checkpoint: {result.error}")
    if result.global_step != BASE_GLOBAL_STEP:
        raise RuntimeError("base checkpoint global step mismatch")
    if result.model_config != "vasu_60m" or not result.finite_tensors:
        raise RuntimeError("base checkpoint is not a finite VASU-60M state")
    if result.sha256 != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("base checkpoint SHA-256 mismatch")


def validate_release(manifest: dict[str, Any]) -> int:
    for path in (
        BASE_CHECKPOINT,
        TOKENIZER_FILE,
        DATA_FILE,
        MASK_FILE,
        RELEASE_MANIFEST_FILE,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(TOKENIZER_FILE) != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    if sha256(DATA_FILE) != manifest["token_sha256"]:
        raise ValueError("token SHA-256 mismatch")
    if sha256(MASK_FILE) != manifest["mask_sha256"]:
        raise ValueError("mask SHA-256 mismatch")

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_FILE))
    if tokenizer.tokenizer.get_vocab_size() != 32_000:
        raise ValueError("tokenizer vocabulary must be 32,000")

    tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode="r")
    mask = np.memmap(MASK_FILE, dtype=np.uint8, mode="r")
    if len(tokens) != len(mask) or len(tokens) % RECORD_LENGTH:
        raise ValueError("invalid fixed-record token/mask files")
    if len(tokens) // RECORD_LENGTH != EXPECTED_RECORDS:
        raise ValueError("packed-record count mismatch")
    mask_records = np.asarray(mask).reshape(EXPECTED_RECORDS, RECORD_LENGTH)
    supervised_tokens = int(mask_records[:, 1:].sum())

    expected_supervised = int(manifest["assistant_loss_tokens"])
    if int(manifest["eos_tokens"]) != 500:
        raise ValueError("expected exactly one supervised EOS token per example")

    if supervised_tokens != expected_supervised:
        raise ValueError(
            "supervised-token count mismatch: "
            f"manifest={expected_supervised}, files={supervised_tokens}"
        )

    return supervised_tokens


def build_train_config() -> TrainConfig:
    return TrainConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        grad_clip=GRAD_CLIP,
        use_amp=USE_AMP,
        checkpoint_path=str(MAIN_CHECKPOINT),
        checkpoint_dir=str(CHECKPOINT_DIR),
        save_every_steps=SAVE_EVERY_STEPS,
    )


def build_datasets() -> tuple[PackedInstructionDataset, PackedInstructionDataset]:
    config = get_vasu_60m_config()
    train_dataset = PackedInstructionDataset(
        str(DATA_FILE), str(MASK_FILE), config.max_seq_len,
        start_record=0, end_record=TRAIN_RECORDS,
    )
    validation_dataset = PackedInstructionDataset(
        str(DATA_FILE), str(MASK_FILE), config.max_seq_len,
        start_record=TRAIN_RECORDS, end_record=EXPECTED_RECORDS,
    )
    if len(train_dataset) != TRAIN_RECORDS:
        raise RuntimeError("unexpected train dataset length")
    if len(validation_dataset) != VALIDATION_RECORDS:
        raise RuntimeError("unexpected validation dataset length")
    return train_dataset, validation_dataset


def configure_engine() -> None:
    validate_output_isolation()
    engine.BASE_CHECKPOINT = BASE_CHECKPOINT
    engine.BASE_GLOBAL_STEP = BASE_GLOBAL_STEP
    engine.TOKENIZER_FILE = TOKENIZER_FILE
    engine.DATA_FILE = DATA_FILE
    engine.MASK_FILE = MASK_FILE
    engine.CHECKPOINT_DIR = CHECKPOINT_DIR
    engine.MAIN_CHECKPOINT = MAIN_CHECKPOINT
    engine.BEST_CHECKPOINT = BEST_CHECKPOINT
    engine.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
    engine.EPOCHS = EPOCHS
    engine.BATCH_SIZE = BATCH_SIZE
    engine.GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS
    engine.LEARNING_RATE = LEARNING_RATE
    engine.WEIGHT_DECAY = WEIGHT_DECAY
    engine.GRAD_CLIP = GRAD_CLIP
    engine.USE_AMP = USE_AMP
    engine.SEED = SEED
    engine.SAVE_EVERY_STEPS = SAVE_EVERY_STEPS


def write_sidecar(path: Path, global_step: int, supervised_tokens: int) -> None:
    manifest = load_manifest()
    atomic_write_json(
        path.with_suffix(".metadata.json"),
        {
            "experiment": EXPERIMENT_NAME,
            "checkpoint": str(path),
            "global_step": global_step,
            "base_checkpoint": str(BASE_CHECKPOINT),
            "base_global_step": BASE_GLOBAL_STEP,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "release_manifest": str(RELEASE_MANIFEST_FILE),
            "release_version": manifest["release_version"],
            "source_sha256": manifest["source_sha256"],
            "review_sha256": manifest["review_sha256"],
            "dataset_token_sha256": manifest["token_sha256"],
            "dataset_mask_sha256": manifest["mask_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "train_packed_records": TRAIN_RECORDS,
            "validation_packed_records": VALIDATION_RECORDS,
            "supervised_tokens_processed": supervised_tokens,
            "loss_definition": "assistant-response and terminating-EOS tokens only",
        },
    )


def output_snapshot() -> set[Path]:
    return set(CHECKPOINT_DIR.rglob("*")) if CHECKPOINT_DIR.exists() else set()


def run_dry_run() -> None:
    before = output_snapshot()
    manifest = load_manifest()
    supervised_tokens = validate_release(manifest)
    validate_base_checkpoint()
    validate_output_isolation()
    engine.ensure_no_fineweb_training_process()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_vasu_60m_config()
    model = VASUModel(config)
    checkpoint = torch.load(
        BASE_CHECKPOINT, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    if engine.count_parameters(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("unexpected parameter count")

    optimizer = build_optimizer(model, build_train_config())
    train_dataset, _ = build_datasets()
    batch = next(iter(DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=True, num_workers=0,
    )))
    inputs, targets, mask = (tensor.to(device) for tensor in batch)
    batch_supervised = int(mask.sum().item())
    if batch_supervised <= 0:
        raise RuntimeError("dry-run batch has zero supervised tokens")
    with torch.no_grad(), torch.amp.autocast(
        "cuda", enabled=device.type == "cuda" and USE_AMP
    ):
        loss = language_model_loss(model(inputs), targets, mask)
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run loss is not finite")
    if optimizer.state:
        raise RuntimeError("dry run changed optimizer state")
    if output_snapshot() != before:
        raise RuntimeError("dry run wrote checkpoint output")

    peak_mib = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda" else None
    )
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print(f"Base global step: {BASE_GLOBAL_STEP}")
    print(f"Parameters: {EXPECTED_PARAMETERS:,}")
    print(f"Train records: {TRAIN_RECORDS}")
    print(f"Validation records: {VALIDATION_RECORDS}")
    print(f"Dataset supervised tokens: {supervised_tokens:,}")
    print(f"Batch shapes: {tuple(inputs.shape)}, {tuple(mask.shape)}")
    print(f"Batch supervised tokens: {batch_supervised:,}")
    print(f"Forward loss: {loss.item():.6f}")
    print(f"Calculated optimizer steps: {optimizer_steps()}")
    print(f"Calculated target global step: {target_global_step()}")
    print(f"Peak CUDA memory: {peak_mib if peak_mib is not None else 'N/A'} MiB")
    print("Checkpoint writes: none")
    print("Training started: False")


def run_training() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("VASU-60M instruction-quality training requires CUDA")
    manifest = load_manifest()
    validate_release(manifest)
    validate_base_checkpoint()
    configure_engine()
    engine.ensure_no_fineweb_training_process()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    available_disk = engine.free_disk_gib()
    if available_disk < engine.MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"At least {engine.MIN_FREE_DISK_GIB} GiB free disk is required; "
            f"found {available_disk:.2f} GiB"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    config = get_vasu_60m_config()
    train_config = build_train_config()
    train_dataset, validation_dataset = build_datasets()

    model = VASUModel(config).to(device)
    if engine.count_parameters(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("unexpected parameter count")
    optimizer = build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    resume = engine.find_latest_checkpoint()
    if resume is None:
        starting_checkpoint = BASE_CHECKPOINT
        checkpoint = engine.load_valid_checkpoint(BASE_CHECKPOINT, "cpu")
        if checkpoint is None:
            raise RuntimeError("invalid base checkpoint")
        model.load_state_dict(checkpoint["model"], strict=True)
        starting_global_step = BASE_GLOBAL_STEP
        prior_average_loss = 0.0
    else:
        starting_checkpoint, checkpoint = resume
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        starting_global_step = int(checkpoint["global_step"])
        prior_average_loss = float(checkpoint["loss"])
        if not BASE_GLOBAL_STEP <= starting_global_step <= target_global_step():
            raise RuntimeError("invalid resume step")
    del checkpoint
    gc.collect()

    completed_steps = starting_global_step - BASE_GLOBAL_STEP
    processed_records = min(
        completed_steps * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        len(train_dataset),
    )
    generator = torch.Generator().manual_seed(SEED)
    shuffled = torch.randperm(len(train_dataset), generator=generator).tolist()
    completed_indices = shuffled[:processed_records]
    remaining_indices = shuffled[processed_records:]
    prior_supervised = engine.count_supervised_tokens(
        train_dataset, completed_indices
    )
    cumulative_loss_sum = prior_average_loss * prior_supervised
    supervised_processed = prior_supervised

    train_loader = DataLoader(
        Subset(train_dataset, remaining_indices), batch_size=BATCH_SIZE,
        shuffle=False, drop_last=True, pin_memory=True, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=0,
    )

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Starting global step: {starting_global_step}")
    print(f"Target global step: {target_global_step()}")
    print(f"Train records: {len(train_dataset)}")
    print(f"Validation records: {len(validation_dataset)}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Free disk before training: {available_disk:.2f} GiB")

    if starting_global_step >= target_global_step():
        print("The experiment is already complete.")
        print(f"Best checkpoint path: {BEST_CHECKPOINT}")
        return

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = starting_global_step
    micro_batches = 0
    latest_loss = prior_average_loss
    maximum_temperature = engine.read_gpu_temperature()
    thermal_stop = False
    progress = tqdm(
        total=target_global_step(), initial=starting_global_step,
        desc="Instruction-quality optimizer steps",
    )

    for batch_index, (inputs, targets, mask) in enumerate(train_loader):
        if global_step >= target_global_step():
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        supervised_count = int(mask.sum().item())
        if supervised_count == 0:
            raise RuntimeError("training batch has zero supervised tokens")
        accumulation_group_start = (
            batch_index // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS
        accumulation_group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            len(train_loader) - accumulation_group_start,
        )

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(inputs)
            raw_loss = language_model_loss(logits, targets, mask)
            if not torch.isfinite(raw_loss):
                raise RuntimeError("non-finite training loss")
            loss = raw_loss / accumulation_group_size
        scaler.scale(loss).backward()
        cumulative_loss_sum += raw_loss.item() * supervised_count
        supervised_processed += supervised_count
        micro_batches += 1
        boundary = (
            micro_batches % GRADIENT_ACCUMULATION_STEPS == 0
            or batch_index + 1 == len(train_loader)
        )
        if not boundary:
            continue
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        latest_loss = cumulative_loss_sum / supervised_processed
        progress.update(1)
        progress.set_postfix(loss=f"{raw_loss.item():.4f}")

        temperature = engine.read_gpu_temperature()
        if temperature is not None:
            maximum_temperature = max(maximum_temperature or temperature, temperature)
        periodic = CHECKPOINT_DIR / f"step_{global_step}.pt"
        for path in (periodic, MAIN_CHECKPOINT):
            engine.atomic_save_checkpoint(
                model, optimizer, scheduler, latest_loss, global_step, path
            )
            write_sidecar(path, global_step, supervised_processed)
        engine.apply_checkpoint_retention()

        if temperature is not None and temperature >= engine.THERMAL_STOP_C:
            thermal_stop = True
            thermal = CHECKPOINT_DIR / f"thermal_stop_step_{global_step}.pt"
            for path in (thermal, MAIN_CHECKPOINT):
                engine.atomic_save_checkpoint(
                    model, optimizer, scheduler, latest_loss, global_step, path
                )
                write_sidecar(path, global_step, supervised_processed)
            engine.apply_checkpoint_retention()
            print(f"Thermal stop at step {global_step}: {temperature} C")
            break

    progress.close()
    if thermal_stop or global_step < target_global_step():
        print(f"Final global step: {global_step}")
        print(f"Train supervised-token loss: {latest_loss:.6f}")
        print("Validation supervised-token loss: not run; epoch incomplete")
        print("Resume supported: True")
        return

    validation_loss, validation_tokens = engine.validate(
        model, validation_loader, device
    )
    scheduler.step()
    for path in (BEST_CHECKPOINT, MAIN_CHECKPOINT):
        engine.atomic_save_checkpoint(
            model, optimizer, scheduler, validation_loss, global_step, path
        )
        write_sidecar(path, global_step, supervised_processed)
    engine.apply_checkpoint_retention()

    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    stable = maximum_temperature is None or maximum_temperature < engine.THERMAL_STOP_C
    print(f"Final global step: {global_step}")
    print(f"Train supervised-token loss: {latest_loss:.6f}")
    print(f"Validation supervised-token loss: {validation_loss:.6f}")
    print(f"Validation supervised tokens: {validation_tokens:,}")
    print(f"Best checkpoint path: {BEST_CHECKPOINT}")
    print(f"Peak CUDA memory: {peak_mib:.2f} MiB")
    print(f"Maximum GPU temperature: {maximum_temperature} C")
    print(f"Training stable: {stable}")
    print("Resume supported: True")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.train:
        parser.error("choose exactly one of --dry-run or --train")
    if args.dry_run:
        run_dry_run()
    else:
        run_training()


if __name__ == "__main__":
    main()
