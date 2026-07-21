"""Run one bounded VASU-60M optimizer update on real FineWeb data."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.checkpoint import save_checkpoint
from vasu.training.dataset import TextDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer


DATA_FILE = "data/processed/pretrain/fineweb_1m.bin"
TOKENIZER_FILE = "assets/tokenizer.json"
CHECKPOINT_DIR = Path("checkpoints/vasu_60m/tiny_test")
CHECKPOINT_PATH = CHECKPOINT_DIR / "vasu.pt"

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
TRAIN_SAMPLES = 32
VALIDATION_SAMPLES = 4


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_datasets(seq_len: int) -> tuple[TextDataset, TextDataset]:
    # TextDataset length is (slice_tokens - seq_len) // seq_len.
    train_end = (TRAIN_SAMPLES + 1) * seq_len
    validation_end = train_end + (VALIDATION_SAMPLES + 1) * seq_len

    train_dataset = TextDataset(
        data_file=DATA_FILE,
        seq_len=seq_len,
        start=0,
        end=train_end,
    )
    validation_dataset = TextDataset(
        data_file=DATA_FILE,
        seq_len=seq_len,
        start=train_end,
        end=validation_end,
    )
    return train_dataset, validation_dataset


def train_one_update(
    model: VASUModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for micro_step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=True):
            logits = model(inputs)
            raw_loss = language_model_loss(logits, targets)
            loss = raw_loss / GRADIENT_ACCUMULATION_STEPS

        scaler.scale(loss).backward()
        total_loss += raw_loss.item()

        if micro_step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: VASUModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True):
            logits = model(inputs)
            loss = language_model_loss(logits, targets)
        total_loss += loss.item()

    return total_loss / len(loader)


def verify_resume(
    model_config,
    train_config: TrainConfig,
    device: torch.device,
) -> None:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    resumed_model = VASUModel(model_config).to(device)
    resumed_optimizer = build_optimizer(resumed_model, train_config)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer,
        T_max=train_config.epochs,
    )

    resumed_model.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])

    if checkpoint.get("global_step") != 1:
        raise AssertionError("Tiny checkpoint did not restore global_step=1.")

    print("Resume verification: OK (fresh model, optimizer, and scheduler)")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This real-data VASU-60M test requires CUDA. Run it from the "
            "CUDA-enabled project virtual environment."
        )
    if CHECKPOINT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing tiny-test checkpoint: "
            f"{CHECKPOINT_PATH}"
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
        checkpoint_path=str(CHECKPOINT_PATH),
        checkpoint_dir=str(CHECKPOINT_DIR),
    )

    tokenizer = VASUTokenizer()
    tokenizer.load(TOKENIZER_FILE)

    train_dataset, validation_dataset = make_datasets(
        model_config.max_seq_len
    )
    train_loader = DataLoader(
        train_dataset,
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

    model = VASUModel(model_config).to(device)
    optimizer = build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_config.epochs,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.use_amp)

    print(f"Parameters: {count_parameters(model):,}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(validation_dataset):,}")

    train_loss = train_one_update(
        model,
        train_loader,
        optimizer,
        scaler,
        device,
    )
    validation_loss = validate(model, validation_loader, device)
    scheduler.step()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        loss=validation_loss,
        path=str(CHECKPOINT_PATH),
        global_step=1,
    )
    verify_resume(model_config, train_config, device)

    peak_memory_mib = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Train loss: {train_loss:.6f}")
    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"CUDA peak memory: {peak_memory_mib:.2f} MiB")


if __name__ == "__main__":
    main()
