"""Train one resumable bounded VASU-60M FineWeb block per run."""

import os
from pathlib import Path
import re
import shutil
import subprocess

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.checkpoint import save_checkpoint
from vasu.training.dataset import ManifestTokenDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer


MANIFEST_FILE = "data/processed/pretrain/fineweb_manifest.json"
TOKENIZER_FILE = "assets/tokenizer.json"
WARMUP_CHECKPOINT = Path(
    "checkpoints/vasu_60m/fineweb_warmup/final_step_500.pt"
)
CHECKPOINT_DIR = Path("checkpoints/vasu_60m/fineweb_blocks")

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
BLOCK_STEPS = 200
TARGET_GLOBAL_STEP = 213_100
SAVE_EVERY_STEPS = 10
VALIDATION_SAMPLES = 256
THERMAL_STOP_C = 88
MIN_FREE_DISK_GIB = 10
KEEP_PERIODIC_CHECKPOINTS = 3
KEEP_BLOCK_FINAL_CHECKPOINTS = 2
KEEP_THERMAL_CHECKPOINTS = 1
PRESERVED_BLOCK_FINAL_NAMES = {"block_final_step_150200.pt"}

DELETED_CHECKPOINTS: list[Path] = []


def resolve_target_global_step(starting_global_step: int) -> int:
    """Return the bounded target for this controlled continuation run."""

    if starting_global_step > TARGET_GLOBAL_STEP:
        raise ValueError(
            f"Checkpoint step {starting_global_step} is beyond the hard target "
            f"{TARGET_GLOBAL_STEP}."
        )
    return min(starting_global_step + BLOCK_STEPS, TARGET_GLOBAL_STEP)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def cuda_memory_summary() -> str:
    mib = 1024**2
    return (
        f"allocated={torch.cuda.memory_allocated() / mib:.2f} MiB, "
        f"reserved={torch.cuda.memory_reserved() / mib:.2f} MiB, "
        f"peak={torch.cuda.max_memory_allocated() / mib:.2f} MiB"
    )


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


def checkpoint_step(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    return int(matches[-1]) if matches else -1


def load_valid_checkpoint(path: Path, map_location) -> dict | None:
    """Return None for corrupt or incomplete checkpoint candidates."""
    try:
        checkpoint = torch.load(path, map_location=map_location)
        required_keys = {
            "epoch",
            "global_step",
            "model",
            "optimizer",
            "scheduler",
            "loss",
        }
        if not required_keys.issubset(checkpoint):
            raise ValueError("missing required checkpoint keys")
        return checkpoint
    except (OSError, RuntimeError, EOFError, ValueError) as error:
        print(f"Ignoring invalid checkpoint {path}: {error}")
        return None


def find_latest_valid_block_checkpoint() -> tuple[Path, dict] | None:
    candidates = sorted(
        CHECKPOINT_DIR.glob("*.pt"),
        key=checkpoint_step,
        reverse=True,
    )
    for candidate in candidates:
        checkpoint = load_valid_checkpoint(candidate, "cpu")
        if checkpoint is not None:
            return candidate, checkpoint
    return None


def apply_checkpoint_retention() -> None:
    groups = (
        ("step_*.pt", KEEP_PERIODIC_CHECKPOINTS),
        ("block_final_step_*.pt", KEEP_BLOCK_FINAL_CHECKPOINTS),
        ("thermal_stop_step_*.pt", KEEP_THERMAL_CHECKPOINTS),
    )
    for pattern, keep_count in groups:
        paths = sorted(
            CHECKPOINT_DIR.glob(pattern),
            key=checkpoint_step,
            reverse=True,
        )
        preserved_names = (
            PRESERVED_BLOCK_FINAL_NAMES
            if pattern == "block_final_step_*.pt"
            else set()
        )
        retained_candidates = [
            path for path in paths if path.name not in preserved_names
        ]
        for old_path in retained_candidates[keep_count:]:
            old_path.unlink()
            DELETED_CHECKPOINTS.append(old_path)


def free_disk_gib() -> float:
    return shutil.disk_usage(CHECKPOINT_DIR.parent).free / (1024**3)


def load_resume_state(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, str, Path]:
    block_resume = find_latest_valid_block_checkpoint()
    if block_resume is None:
        if not WARMUP_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Required warmup checkpoint not found: {WARMUP_CHECKPOINT}"
            )
        resume_path = WARMUP_CHECKPOINT
        resume_source = "warmup checkpoint"
        checkpoint = load_valid_checkpoint(resume_path, device)
        if checkpoint is None:
            raise RuntimeError(f"Warmup checkpoint is invalid: {resume_path}")
    else:
        resume_path, checkpoint = block_resume
        resume_source = "block checkpoint"

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    if resume_source == "block checkpoint":
        scheduler.load_state_dict(checkpoint["scheduler"])

    # Block training intentionally uses a constant 1e-4 learning rate. The
    # warmup checkpoint ended its cosine schedule at zero, so restore the
    # requested block-training rate after loading optimizer state.
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = 1e-4

    global_step = int(checkpoint.get("global_step", 0))
    return global_step, resume_source, resume_path


def build_datasets(
    seq_len: int,
    starting_global_step: int,
    target_global_step: int,
) -> tuple[ManifestTokenDataset, ManifestTokenDataset]:
    optimizer_steps = target_global_step - starting_global_step
    if optimizer_steps <= 0 or optimizer_steps > BLOCK_STEPS:
        raise ValueError(
            "Dataset window requires between 1 and "
            f"{BLOCK_STEPS} optimizer steps; found {optimizer_steps}."
        )
    samples_per_optimizer_step = (
        BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    )
    block_samples = optimizer_steps * samples_per_optimizer_step

    # Global step is the authoritative sequential position in the logical
    # [original training slice][extension shard] stream.
    train_start = starting_global_step * samples_per_optimizer_step * seq_len
    train_end = train_start + block_samples * seq_len + 1
    validation_end = VALIDATION_SAMPLES * seq_len + 1

    train_dataset = ManifestTokenDataset(
        manifest_path=MANIFEST_FILE,
        split="train",
        sequence_length=seq_len,
        logical_start=train_start,
        logical_end=train_end,
    )
    validation_dataset = ManifestTokenDataset(
        manifest_path=MANIFEST_FILE,
        split="validation",
        sequence_length=seq_len,
        logical_start=0,
        logical_end=validation_end,
    )
    return train_dataset, validation_dataset


def save_block_checkpoint(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: float,
    global_step: int,
    path: Path,
) -> None:

    if free_disk_gib() < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"Checkpoint save refused: only {free_disk_gib():.2f} GiB free."
        )

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
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
    apply_checkpoint_retention()


@torch.no_grad()
def validate(
    model: VASUModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Validation", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(inputs)
            loss = language_model_loss(logits, targets)
        total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VASU-60M FineWeb block training requires the CUDA-enabled "
            "project virtual environment."
        )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    model_config = get_vasu_60m_config()
    train_config = TrainConfig(
        batch_size=BATCH_SIZE,
        epochs=1,
        learning_rate=1e-4,
        weight_decay=0.1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_amp=True,
        checkpoint_dir=str(CHECKPOINT_DIR),
        checkpoint_path=str(CHECKPOINT_DIR / "unused.pt"),
        save_every_steps=SAVE_EVERY_STEPS,
    )

    tokenizer = VASUTokenizer()
    tokenizer.load(TOKENIZER_FILE)

    model = VASUModel(model_config).to(device)
    optimizer = build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    free_before = free_disk_gib()
    print(f"Free disk before training: {free_before:.2f} GiB")
    if free_before < MIN_FREE_DISK_GIB:
        raise RuntimeError(
            f"At least {MIN_FREE_DISK_GIB} GiB free disk is required; "
            f"found {free_before:.2f} GiB."
        )

    starting_global_step, resume_source, resume_path = load_resume_state(
        model,
        optimizer,
        scheduler,
        device,
    )
    apply_checkpoint_retention()
    target_global_step = resolve_target_global_step(starting_global_step)
    if starting_global_step == target_global_step:
        print(
            f"Hard target {TARGET_GLOBAL_STEP} is already reached; "
            "no training process will be started."
        )
        return

    train_dataset, validation_dataset = build_datasets(
        model_config.max_seq_len,
        starting_global_step,
        target_global_step,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        # Required for exact global-step-to-token progression and resumability.
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
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    print(f"Parameter count: {count_parameters(model):,}")
    print(f"Resume source: {resume_source} ({resume_path})")
    print(f"Starting global step: {starting_global_step}")
    print(f"Target global step: {target_global_step}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(
        "THERMAL WARNING: this run stops and checkpoints automatically at "
        f"{THERMAL_STOP_C} C."
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = starting_global_step
    accumulated_loss = 0.0
    processed_micro_steps = 0
    latest_train_loss = float("nan")
    latest_checkpoint = resume_path
    maximum_temperature = read_gpu_temperature()
    thermal_stop = False

    progress = tqdm(
        total=target_global_step,
        initial=starting_global_step,
        desc="Optimizer steps",
    )

    for inputs, targets in train_loader:
        if global_step >= target_global_step:
            break

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(inputs)
            raw_loss = language_model_loss(logits, targets)
            loss = raw_loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        accumulated_loss += raw_loss.item()
        processed_micro_steps += 1

        if processed_micro_steps % GRADIENT_ACCUMULATION_STEPS != 0:
            continue

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1
        progress.update(1)
        latest_train_loss = accumulated_loss / processed_micro_steps
        progress.set_postfix(loss=f"{raw_loss.item():.4f}")

        temperature = read_gpu_temperature()
        if temperature is not None:
            maximum_temperature = max(
                maximum_temperature or temperature,
                temperature,
            )

        if global_step % SAVE_EVERY_STEPS == 0:
            step_path = CHECKPOINT_DIR / f"step_{global_step}.pt"
            save_block_checkpoint(
                model,
                optimizer,
                scheduler,
                latest_train_loss,
                global_step,
                step_path,
            )
            latest_checkpoint = step_path
            print(
                f"Step {global_step} | loss={latest_train_loss:.6f} | "
                f"temperature={temperature} C | CUDA {cuda_memory_summary()}"
            )

        if temperature is not None and temperature >= THERMAL_STOP_C:
            thermal_stop = True
            thermal_path = (
                CHECKPOINT_DIR / f"thermal_stop_step_{global_step}.pt"
            )
            save_block_checkpoint(
                model,
                optimizer,
                scheduler,
                latest_train_loss,
                global_step,
                thermal_path,
            )
            latest_checkpoint = thermal_path
            print(f"Thermal stop at step {global_step}: {temperature} C")
            break

    progress.close()

    block_completed = global_step >= target_global_step
    crossed_validation_milestone = (
        global_step // 250 > starting_global_step // 250
    )

    if crossed_validation_milestone:
        validation_loss = validate(model, validation_loader, device)
        print(f"Validation milestone reached at step {global_step}.")
    else:
        validation_loss = float("nan")
        print("Validation skipped; no 250-step milestone was crossed.")

    if block_completed:
        final_path = CHECKPOINT_DIR / f"block_final_step_{global_step}.pt"
        checkpoint_loss = (
            validation_loss
            if crossed_validation_milestone
            else latest_train_loss
        )
        save_block_checkpoint(
            model,
            optimizer,
            scheduler,
            checkpoint_loss,
            global_step,
            final_path,
        )
        latest_checkpoint = final_path

    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Resume checkpoint: {resume_path}")
    print(f"Starting global step: {starting_global_step}")
    print(f"Target global step: {target_global_step}")
    print(f"Final global step: {global_step}")
    print(f"Latest loss: {latest_train_loss:.6f}")
    print(f"Block completed: {block_completed}")
    print(f"Thermal stop occurred: {thermal_stop}")
    if crossed_validation_milestone:
        print(f"Validation loss: {validation_loss:.6f}")
    print(f"Latest checkpoint path: {latest_checkpoint}")
    print(f"CUDA peak memory: {peak_memory_mib:.2f} MiB")
    print(f"Maximum GPU temperature: {maximum_temperature} C")
    print(f"Free disk after training: {free_disk_gib():.2f} GiB")
    print("Retained checkpoints:")
    for checkpoint_path in sorted(
        CHECKPOINT_DIR.glob("*.pt"),
        key=lambda path: (checkpoint_step(path), path.name),
    ):
        print(f"  {checkpoint_path}")
    print("Deleted checkpoints:")
    if DELETED_CHECKPOINTS:
        for checkpoint_path in DELETED_CHECKPOINTS:
            print(f"  {checkpoint_path}")
    else:
        print("  none")


if __name__ == "__main__":
    main()

