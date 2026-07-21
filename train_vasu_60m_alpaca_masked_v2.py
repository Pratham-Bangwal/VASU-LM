"""Run one resumable assistant-only Alpaca-v2 epoch for VASU-60M."""

import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.data.alpaca_masked_v2 import (
    FORMAT_VERSION,
    atomic_write_json,
)
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.checkpoint import save_checkpoint
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer


BASE_CHECKPOINT = Path(
    "checkpoints/vasu_60m/milestones/fineweb_step_150000.pt"
)
TOKENIZER_FILE = Path("assets/tokenizer.json")
DATA_FILE = Path("data/processed/instruct/alpaca_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/alpaca_masked_v2_mask.bin")
DATASET_METADATA_FILE = Path(
    "data/processed/instruct/alpaca_masked_v2_metadata.json"
)
CHECKPOINT_DIR = Path("checkpoints/vasu_60m/alpaca_masked_v2")
MAIN_CHECKPOINT = CHECKPOINT_DIR / "vasu.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

BASE_GLOBAL_STEP = 150_000
EXPECTED_PARAMETERS = 58_337_792
EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_AMP = True
SEED = 42

SAVE_EVERY_STEPS = 100
KEEP_PERIODIC_CHECKPOINTS = 2
KEEP_THERMAL_CHECKPOINTS = 1
THERMAL_STOP_C = 88
MIN_FREE_DISK_GIB = 10

REQUIRED_CHECKPOINT_KEYS = {
    "epoch",
    "global_step",
    "model",
    "optimizer",
    "scheduler",
    "loss",
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def free_disk_gib() -> float:
    return shutil.disk_usage(CHECKPOINT_DIR.parent).free / (1024**3)


def read_gpu_temperature() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def ensure_no_fineweb_training_process() -> None:
    """Refuse to compete with an active FineWeb trainer or runner."""
    command = (
        "$current = $PID; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -ne $current -and "
        "$_.CommandLine -match "
        "'train_vasu_60m_fineweb_blocks.py|"
        "run_vasu_60m_for_11_hours.ps1|"
        "run_vasu_60m_until_' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    )
    process_ids = [line for line in result.stdout.splitlines() if line.strip()]
    if process_ids:
        raise RuntimeError(
            "Conflicting FineWeb training process detected: "
            + ", ".join(process_ids)
        )


def load_valid_checkpoint(
    path: Path,
    map_location: torch.device | str,
) -> dict | None:
    try:
        checkpoint = torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint payload is not a dictionary")
        if not REQUIRED_CHECKPOINT_KEYS.issubset(checkpoint):
            raise ValueError("checkpoint is missing required keys")
        return checkpoint
    except (OSError, RuntimeError, EOFError, ValueError) as error:
        print(f"Ignoring invalid checkpoint {path}: {error}")
        return None


def find_latest_checkpoint() -> tuple[Path, dict] | None:
    if not CHECKPOINT_DIR.exists():
        return None
    latest: tuple[int, int, Path, dict] | None = None
    for path in CHECKPOINT_DIR.glob("*.pt"):
        if ".tmp" in path.suffixes:
            continue
        checkpoint = load_valid_checkpoint(path, "cpu")
        if checkpoint is None:
            continue
        candidate = (
            int(checkpoint["global_step"]),
            path.stat().st_mtime_ns,
            path,
            checkpoint,
        )
        if latest is None or candidate[:2] > latest[:2]:
            if latest is not None:
                del latest
                gc.collect()
            latest = candidate
        else:
            del checkpoint
            gc.collect()
    if latest is None:
        return None
    _, _, path, checkpoint = latest
    return path, checkpoint


def retain_latest(pattern: str, keep_count: int) -> None:
    paths = sorted(
        CHECKPOINT_DIR.glob(pattern),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for old_path in paths[keep_count:]:
        old_path.unlink()
        old_path.with_suffix(".metadata.json").unlink(missing_ok=True)
        print(f"Deleted retained checkpoint: {old_path}")


def apply_checkpoint_retention() -> None:
    retain_latest("step_*.pt", KEEP_PERIODIC_CHECKPOINTS)
    retain_latest("thermal_stop_step_*.pt", KEEP_THERMAL_CHECKPOINTS)


def atomic_save_checkpoint(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: float,
    global_step: int,
    path: Path,
) -> None:
    available = free_disk_gib()
    if available < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"Checkpoint save refused: only {available:.2f} GiB free."
        )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=0,
            loss=loss,
            path=str(temporary_path),
            global_step=global_step,
        )
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def write_checkpoint_sidecar(
    checkpoint_path: Path,
    global_step: int,
    supervised_tokens_processed: int,
) -> None:
    """Record experiment metadata without changing checkpoint payload keys."""
    sidecar = checkpoint_path.with_suffix(".metadata.json")
    atomic_write_json(
        sidecar,
        {
            "checkpoint": str(checkpoint_path),
            "global_step": global_step,
            "dataset_format_version": FORMAT_VERSION,
            "dataset_metadata": str(DATASET_METADATA_FILE),
            "supervised_tokens_processed": supervised_tokens_processed,
            "loss_definition": "assistant-response and terminating-EOS tokens only",
        },
    )


def count_supervised_tokens(
    dataset: PackedInstructionDataset,
    indices: list[int],
) -> int:
    total = 0
    record_length = dataset.record_length
    for index in indices:
        record_index = dataset.start_record + index
        start = record_index * record_length
        record_mask = dataset.mask[start : start + record_length]
        total += int(record_mask[1:].sum())
    return total


@torch.no_grad()
def validate(
    model: VASUModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    model.eval()
    loss_sum = 0.0
    supervised_tokens = 0
    for inputs, targets, mask in tqdm(loader, desc="Validation", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        token_count = int(mask.sum().item())
        if token_count == 0:
            raise RuntimeError("validation batch has zero supervised tokens")
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(inputs)
            loss = language_model_loss(logits, targets, mask)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite validation loss")
        loss_sum += loss.item() * token_count
        supervised_tokens += token_count
    if supervised_tokens == 0:
        raise RuntimeError("validation contains no supervised tokens")
    return loss_sum / supervised_tokens, supervised_tokens


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("VASU-60M masked Alpaca v2 requires CUDA.")
    ensure_no_fineweb_training_process()

    required_files = (
        BASE_CHECKPOINT,
        TOKENIZER_FILE,
        DATA_FILE,
        MASK_FILE,
        DATASET_METADATA_FILE,
    )
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = json.loads(DATASET_METADATA_FILE.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected masked-Alpaca-v2 dataset format")
    if int(metadata["sequence_length"]) != 256:
        raise ValueError("masked-Alpaca-v2 sequence length must be 256")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    available_disk = free_disk_gib()
    if available_disk < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"At least {MIN_FREE_DISK_GIB} GiB free disk is required; "
            f"found {available_disk:.2f} GiB."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    model_config = get_vasu_60m_config()
    train_config = TrainConfig(
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

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_FILE))
    full_dataset = PackedInstructionDataset(
        str(DATA_FILE), str(MASK_FILE), model_config.max_seq_len
    )
    split_record = int(len(full_dataset) * 0.95)
    if split_record <= 0 or split_record >= len(full_dataset):
        raise ValueError("dataset is too small for deterministic train/validation split")
    train_dataset = PackedInstructionDataset(
        str(DATA_FILE),
        str(MASK_FILE),
        model_config.max_seq_len,
        start_record=0,
        end_record=split_record,
    )
    validation_dataset = PackedInstructionDataset(
        str(DATA_FILE),
        str(MASK_FILE),
        model_config.max_seq_len,
        start_record=split_record,
    )

    full_micro_batches = len(train_dataset) // BATCH_SIZE
    optimizer_steps = math.ceil(
        full_micro_batches / GRADIENT_ACCUMULATION_STEPS
    )
    target_global_step = BASE_GLOBAL_STEP + optimizer_steps

    model = VASUModel(model_config).to(device)
    parameter_count = count_parameters(model)
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count:,}.")
    optimizer = build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    resume = find_latest_checkpoint()
    if resume is None:
        starting_checkpoint = BASE_CHECKPOINT
        checkpoint = load_valid_checkpoint(BASE_CHECKPOINT, "cpu")
        if checkpoint is None:
            raise RuntimeError(f"Invalid base checkpoint: {BASE_CHECKPOINT}")
        if int(checkpoint["global_step"]) != BASE_GLOBAL_STEP:
            raise RuntimeError("base checkpoint global_step is not 150000")
        model.load_state_dict(checkpoint["model"])
        starting_global_step = BASE_GLOBAL_STEP
        prior_average_loss = 0.0
    else:
        starting_checkpoint, checkpoint = resume
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        starting_global_step = int(checkpoint["global_step"])
        prior_average_loss = float(checkpoint["loss"])
        if not BASE_GLOBAL_STEP <= starting_global_step <= target_global_step:
            raise RuntimeError(
                f"invalid masked-v2 resume step: {starting_global_step}"
            )
    del checkpoint
    gc.collect()

    completed_steps = starting_global_step - BASE_GLOBAL_STEP
    processed_records = min(
        completed_steps * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        len(train_dataset),
    )
    generator = torch.Generator().manual_seed(SEED)
    shuffled_indices = torch.randperm(
        len(train_dataset), generator=generator
    ).tolist()
    completed_indices = shuffled_indices[:processed_records]
    remaining_indices = shuffled_indices[processed_records:]
    prior_supervised_tokens = count_supervised_tokens(
        train_dataset, completed_indices
    )
    cumulative_loss_sum = prior_average_loss * prior_supervised_tokens
    supervised_tokens_processed = prior_supervised_tokens
    remaining_dataset = Subset(train_dataset, remaining_indices)

    train_loader = DataLoader(
        remaining_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )

    print(f"Device: {device}")
    print(f"Parameter count: {parameter_count:,}")
    print(f"Dataset format: {metadata['format_version']}")
    print(f"Starting checkpoint: {starting_checkpoint}")
    print(f"Starting global step: {starting_global_step}")
    print(f"Target global step: {target_global_step}")
    print(f"Train records: {len(train_dataset):,}")
    print(f"Remaining train records: {len(remaining_dataset):,}")
    print(f"Validation records: {len(validation_dataset):,}")
    print(f"Prior supervised tokens: {prior_supervised_tokens:,}")
    print(
        "Optimizer settings: "
        f"AdamW lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}, "
        f"batch_size={BATCH_SIZE}, accumulation="
        f"{GRADIENT_ACCUMULATION_STEPS}, grad_clip={GRAD_CLIP}, AMP={USE_AMP}"
    )
    print(f"Free disk before training: {available_disk:.2f} GiB")
    print(f"THERMAL WARNING: training stops and saves at {THERMAL_STOP_C} C.")

    if starting_global_step >= target_global_step:
        print("The one-epoch masked Alpaca v2 experiment is already complete.")
        print(f"Best checkpoint path: {BEST_CHECKPOINT}")
        return

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = starting_global_step
    micro_batches = 0
    latest_train_loss = prior_average_loss
    maximum_temperature = read_gpu_temperature()
    thermal_stop = False
    zero_supervision_batches = 0
    progress = tqdm(
        total=target_global_step,
        initial=starting_global_step,
        desc="Masked Alpaca v2 optimizer steps",
    )

    for batch_index, (inputs, targets, mask) in enumerate(train_loader):
        if global_step >= target_global_step:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        supervised_count = int(mask.sum().item())
        if supervised_count == 0:
            zero_supervision_batches += 1
            raise RuntimeError("training batch has zero supervised tokens")

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(inputs)
            raw_loss = language_model_loss(logits, targets, mask)
            if not torch.isfinite(raw_loss):
                raise RuntimeError("non-finite masked training loss")
            loss = raw_loss / GRADIENT_ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        cumulative_loss_sum += raw_loss.item() * supervised_count
        supervised_tokens_processed += supervised_count
        micro_batches += 1

        optimizer_boundary = (
            micro_batches % GRADIENT_ACCUMULATION_STEPS == 0
            or batch_index + 1 == len(train_loader)
        )
        if not optimizer_boundary:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        latest_train_loss = cumulative_loss_sum / supervised_tokens_processed
        progress.update(1)
        progress.set_postfix(loss=f"{raw_loss.item():.4f}")

        temperature = read_gpu_temperature()
        if temperature is not None:
            maximum_temperature = max(
                maximum_temperature or temperature, temperature
            )

        if global_step % SAVE_EVERY_STEPS == 0:
            periodic_path = CHECKPOINT_DIR / f"step_{global_step}.pt"
            for path in (periodic_path, MAIN_CHECKPOINT):
                atomic_save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    latest_train_loss,
                    global_step,
                    path,
                )
                write_checkpoint_sidecar(
                    path, global_step, supervised_tokens_processed
                )
            apply_checkpoint_retention()
            print(
                f"Step {global_step} | assistant_loss={latest_train_loss:.6f} "
                f"| supervised_tokens={supervised_tokens_processed:,} "
                f"| temperature={temperature} C"
            )

        if temperature is not None and temperature >= THERMAL_STOP_C:
            thermal_stop = True
            thermal_path = (
                CHECKPOINT_DIR / f"thermal_stop_step_{global_step}.pt"
            )
            for path in (thermal_path, MAIN_CHECKPOINT):
                atomic_save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    latest_train_loss,
                    global_step,
                    path,
                )
                write_checkpoint_sidecar(
                    path, global_step, supervised_tokens_processed
                )
            apply_checkpoint_retention()
            print(f"Thermal stop at step {global_step}: {temperature} C")
            break

    progress.close()
    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024**2)
    if thermal_stop or global_step < target_global_step:
        print(f"Final global step: {global_step}")
        print(f"Train supervised-token loss: {latest_train_loss:.6f}")
        print("Validation supervised-token loss: not run; epoch incomplete")
        print(f"Supervised tokens processed: {supervised_tokens_processed:,}")
        print(f"Zero-supervision batches: {zero_supervision_batches}")
        print(f"Peak CUDA memory: {peak_memory_mib:.2f} MiB")
        print(f"Maximum GPU temperature: {maximum_temperature} C")
        print("Resume supported: True")
        return

    validation_loss, validation_supervised_tokens = validate(
        model, validation_loader, device
    )
    scheduler.step()
    for path in (BEST_CHECKPOINT, MAIN_CHECKPOINT):
        atomic_save_checkpoint(
            model,
            optimizer,
            scheduler,
            validation_loss,
            global_step,
            path,
        )
        write_checkpoint_sidecar(
            path, global_step, supervised_tokens_processed
        )
    apply_checkpoint_retention()

    thermally_stable = (
        maximum_temperature is None or maximum_temperature < THERMAL_STOP_C
    )
    print(f"Final global step: {global_step}")
    print(f"Train supervised-token loss: {latest_train_loss:.6f}")
    print(f"Validation supervised-token loss: {validation_loss:.6f}")
    print(f"Validation supervised tokens: {validation_supervised_tokens:,}")
    print(f"Supervised tokens processed: {supervised_tokens_processed:,}")
    print(f"Zero-supervision batches: {zero_supervision_batches}")
    print(f"Best checkpoint path: {BEST_CHECKPOINT}")
    print(f"Main resumable checkpoint: {MAIN_CHECKPOINT}")
    print(f"Peak CUDA memory: {peak_memory_mib:.2f} MiB")
    print(f"Maximum GPU temperature: {maximum_temperature} C")
    print(f"Training stable: {thermally_stable}")
    print("Resume supported: True")


if __name__ == "__main__":
    main()
