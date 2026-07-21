"""Controlled 500-step FineWeb warmup for the opt-in VASU-60M model."""

import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.checkpoint import save_checkpoint
from vasu.training.dataset import TextDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer


DATA_FILE = "data/processed/pretrain/fineweb_1m.bin"
TOKENIZER_FILE = "assets/tokenizer.json"
CHECKPOINT_DIR = Path("checkpoints/vasu_60m/fineweb_warmup")

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
MAX_OPTIMIZER_STEPS = 500
SAVE_EVERY_STEPS = 25
VALIDATION_SAMPLES = 256
THERMAL_STOP_C = 88
FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR / f"final_step_{MAX_OPTIMIZER_STEPS}.pt"
)


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


def build_datasets(seq_len: int) -> tuple[TextDataset, TextDataset]:
    total_tokens = len(np.memmap(DATA_FILE, dtype=np.uint16, mode="r"))
    train_samples = (
        MAX_OPTIMIZER_STEPS
        * GRADIENT_ACCUMULATION_STEPS
        * BATCH_SIZE
    )

    train_end = (train_samples + 1) * seq_len
    validation_start = int(total_tokens * 0.98)
    validation_end = validation_start + (VALIDATION_SAMPLES + 1) * seq_len

    if validation_end > total_tokens:
        raise ValueError("FineWeb file is too small for the validation slice.")

    train_dataset = TextDataset(
        data_file=DATA_FILE,
        seq_len=seq_len,
        start=0,
        end=train_end,
    )
    validation_dataset = TextDataset(
        data_file=DATA_FILE,
        seq_len=seq_len,
        start=validation_start,
        end=validation_end,
    )
    return train_dataset, validation_dataset


def find_latest_checkpoint() -> Path | None:
    candidates = list(CHECKPOINT_DIR.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resume_warmup(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    latest = find_latest_checkpoint()
    if latest is None:
        print("Starting VASU-60M FineWeb warmup from scratch.")
        return 0

    checkpoint = torch.load(latest, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = int(checkpoint.get("global_step", 0))

    # The first controlled run used a 100-step cosine horizon. When extending
    # it, place the scheduler at the equivalent point on the new horizon so
    # training does not resume with the old schedule's zero learning rate.
    if scheduler.T_max != MAX_OPTIMIZER_STEPS:
        scheduler.T_max = MAX_OPTIMIZER_STEPS
        scheduler.last_epoch = global_step
        factor = 0.5 * (
            1.0 + math.cos(math.pi * global_step / MAX_OPTIMIZER_STEPS)
        )
        resumed_lrs = [base_lr * factor for base_lr in scheduler.base_lrs]
        for parameter_group, learning_rate in zip(
            optimizer.param_groups,
            resumed_lrs,
        ):
            parameter_group["lr"] = learning_rate
        scheduler._last_lr = resumed_lrs

    print(f"Resumed warmup from {latest} at global_step={global_step}.")
    return global_step


def save_warmup_checkpoint(
    model: VASUModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss: float,
    global_step: int,
    path: Path,
) -> None:
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        loss=loss,
        path=str(path),
        global_step=global_step,
    )


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
            "VASU-60M FineWeb warmup requires the CUDA-enabled project "
            "virtual environment."
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
        checkpoint_path=str(FINAL_CHECKPOINT_PATH),
        checkpoint_dir=str(CHECKPOINT_DIR),
        save_every_steps=SAVE_EVERY_STEPS,
    )

    tokenizer = VASUTokenizer()
    tokenizer.load(TOKENIZER_FILE)

    train_dataset, validation_dataset = build_datasets(
        model_config.max_seq_len
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    model = VASUModel(model_config).to(device)
    optimizer = build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_OPTIMIZER_STEPS,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    global_step = resume_warmup(model, optimizer, scheduler, device)
    starting_global_step = global_step

    print(f"Starting global step: {starting_global_step}")
    print(f"Parameter count: {count_parameters(model):,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")
    print(f"Warmup target optimizer steps: {MAX_OPTIMIZER_STEPS:,}")
    print(
        "THERMAL WARNING: monitor GPU temperature and stop the run manually "
        "if it reaches 88-90 C. Checkpoints are saved every 25 steps."
    )

    maximum_temperature: int | None = read_gpu_temperature()
    latest_checkpoint: Path | None = find_latest_checkpoint()
    thermal_stop = False

    if global_step >= MAX_OPTIMIZER_STEPS:
        print("Warmup optimizer-step target already reached; validating only.")
        final_train_loss = float("nan")
    else:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        processed_micro_steps = 0
        progress = tqdm(
            total=MAX_OPTIMIZER_STEPS,
            initial=global_step,
            desc="Optimizer steps",
        )

        for inputs, targets in train_loader:
            if global_step >= MAX_OPTIMIZER_STEPS:
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
            current_train_loss = accumulated_loss / processed_micro_steps
            progress.set_postfix(loss=f"{raw_loss.item():.4f}")

            temperature = read_gpu_temperature()
            if temperature is not None:
                maximum_temperature = max(
                    maximum_temperature or temperature,
                    temperature,
                )

            if global_step % SAVE_EVERY_STEPS == 0:
                step_path = CHECKPOINT_DIR / f"step_{global_step}.pt"
                save_warmup_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    current_train_loss,
                    global_step,
                    step_path,
                )
                latest_checkpoint = step_path
                print(f"Saved checkpoint: {step_path}")
                print(
                    f"Step {global_step} | loss={current_train_loss:.6f} | "
                    f"temperature={temperature} C | "
                    f"CUDA {cuda_memory_summary()}"
                )

            if temperature is not None and temperature >= THERMAL_STOP_C:
                thermal_stop = True
                if global_step % SAVE_EVERY_STEPS != 0:
                    thermal_path = (
                        CHECKPOINT_DIR / f"thermal_stop_step_{global_step}.pt"
                    )
                    save_warmup_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        current_train_loss,
                        global_step,
                        thermal_path,
                    )
                    latest_checkpoint = thermal_path
                print(
                    f"Thermal stop at step {global_step}: "
                    f"GPU reached {temperature} C."
                )
                break

        progress.close()
        final_train_loss = accumulated_loss / processed_micro_steps

    if global_step >= MAX_OPTIMIZER_STEPS:
        final_validation_loss = validate(model, validation_loader, device)
        if not FINAL_CHECKPOINT_PATH.exists():
            save_warmup_checkpoint(
                model,
                optimizer,
                scheduler,
                final_validation_loss,
                global_step,
                FINAL_CHECKPOINT_PATH,
            )
            latest_checkpoint = FINAL_CHECKPOINT_PATH
        else:
            print(
                "Final checkpoint already exists; refusing to overwrite: "
                f"{FINAL_CHECKPOINT_PATH}"
            )
    else:
        final_validation_loss = float("nan")
        print("Validation skipped because step 500 was not reached.")

    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Global step: {global_step}")
    print(f"Final train loss: {final_train_loss:.6f}")
    print(f"Final validation loss: {final_validation_loss:.6f}")
    print(f"CUDA peak memory: {peak_memory_mib:.2f} MiB")
    print(f"Latest checkpoint path: {latest_checkpoint}")
    print(f"Maximum GPU temperature: {maximum_temperature} C")
    print(
        "Remained below thermal threshold: "
        f"{not thermal_stop and (maximum_temperature or 0) < THERMAL_STOP_C}"
    )


if __name__ == "__main__":
    main()
