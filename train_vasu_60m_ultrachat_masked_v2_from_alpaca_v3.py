"""One-epoch UltraChat masked-v2 continuation from masked Alpaca v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_vasu_60m_alpaca_masked_v2 as engine
from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.data.alpaca_masked_v2 import atomic_write_json
from vasu.data.ultrachat_masked_v2 import FORMAT_VERSION, validate_records
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer
from vasu.training.orchestration import inspect_checkpoint


EXPERIMENT_NAME = "vasu_60m_ultrachat_masked_v2_from_alpaca_v3"
BASE_CHECKPOINT = Path(
    "checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt"
)
BASE_GLOBAL_STEP = 200_711
BASE_CHECKPOINT_SHA256 = (
    "c5da8e1f95f84ad391338548ab777d2aabf3f931f7f6c5aff54c040caef63c43"
)
TOKENIZER_FILE = Path("assets/tokenizer.json")
DATA_FILE = Path("data/processed/instruct/ultrachat_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/ultrachat_masked_v2_mask.bin")
METADATA_FILE = Path(
    "data/processed/instruct/ultrachat_masked_v2_metadata.json"
)
CHECKPOINT_DIR = Path(
    "checkpoints/vasu_60m/ultrachat_masked_v2_from_alpaca_v3"
)
MAIN_CHECKPOINT = CHECKPOINT_DIR / "vasu.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

EXPECTED_PARAMETERS = 58_337_792
EPOCHS = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
USE_AMP = True
SEED = 42
SAVE_EVERY_STEPS = 100

PROTECTED_DIRECTORIES = (
    Path("checkpoints/vasu_60m/alpaca"),
    Path("checkpoints/vasu_60m/alpaca_masked_v2"),
    Path("checkpoints/vasu_60m/alpaca_masked_v3_from_200k"),
    Path("checkpoints/vasu_60m/fineweb_blocks"),
    Path("checkpoints/vasu_60m/milestones"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata() -> dict[str, Any]:
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected UltraChat masked-v2 dataset format")
    if int(metadata.get("sequence_length", -1)) != 256:
        raise ValueError("UltraChat masked-v2 sequence length must be 256")
    if int(metadata.get("record_length", -1)) != 257:
        raise ValueError("UltraChat masked-v2 record length must be 257")
    return metadata


def validate_inputs(metadata: dict[str, Any]) -> tuple[int, int]:
    for path in (BASE_CHECKPOINT, TOKENIZER_FILE, DATA_FILE, MASK_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)
    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_FILE))
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    if vocab_size != 32_000 or vocab_size != int(metadata["vocabulary_size"]):
        raise ValueError("tokenizer/model/dataset vocabulary mismatch")
    if sha256(TOKENIZER_FILE) != metadata["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    if sha256(DATA_FILE) != metadata["token_file_sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    if sha256(MASK_FILE) != metadata["mask_file_sha256"]:
        raise ValueError("mask SHA-256 mismatch")

    record_length = int(metadata["record_length"])
    flat_tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode="r")
    flat_mask = np.memmap(MASK_FILE, dtype=np.uint8, mode="r")
    if len(flat_tokens) != len(flat_mask) or len(flat_tokens) % record_length:
        raise ValueError("invalid fixed-record token/mask files")
    tokens = np.asarray(flat_tokens).reshape(-1, record_length)
    mask = np.asarray(flat_mask).reshape(-1, record_length)
    validate_records(tokens, mask, metadata)
    return len(tokens), int(mask.sum())


def validate_base_checkpoint() -> None:
    inspection = inspect_checkpoint(
        BASE_CHECKPOINT, verify_finite=True, calculate_hash=True
    )
    if not inspection.valid:
        raise RuntimeError(f"invalid Alpaca-v3 base: {inspection.error}")
    if inspection.global_step != BASE_GLOBAL_STEP:
        raise RuntimeError(
            f"base step must be {BASE_GLOBAL_STEP}; found {inspection.global_step}"
        )
    if inspection.model_config != "vasu_60m" or not inspection.finite_tensors:
        raise RuntimeError("base checkpoint is not a finite VASU-60M state")
    if inspection.sha256 != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("Alpaca-v3 base checkpoint SHA-256 mismatch")


def validate_output_isolation() -> None:
    output = CHECKPOINT_DIR.resolve()
    for protected in PROTECTED_DIRECTORIES:
        protected = protected.resolve()
        if output == protected or protected in output.parents:
            raise ValueError(f"output overlaps protected checkpoint path: {protected}")


def write_sidecar(
    checkpoint_path: Path,
    global_step: int,
    supervised_tokens_processed: int,
) -> None:
    if CHECKPOINT_DIR.resolve() not in checkpoint_path.resolve().parents:
        raise ValueError("refusing to write UltraChat sidecar outside output")
    metadata = load_metadata()
    atomic_write_json(
        checkpoint_path.with_suffix(".metadata.json"),
        {
            "experiment": EXPERIMENT_NAME,
            "checkpoint": str(checkpoint_path),
            "global_step": global_step,
            "base_checkpoint": str(BASE_CHECKPOINT),
            "base_global_step": BASE_GLOBAL_STEP,
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "dataset_format_version": FORMAT_VERSION,
            "dataset_metadata": str(METADATA_FILE),
            "dataset_token_sha256": metadata["token_file_sha256"],
            "dataset_mask_sha256": metadata["mask_file_sha256"],
            "tokenizer_sha256": metadata["tokenizer_sha256"],
            "supervised_tokens_processed": supervised_tokens_processed,
            "loss_definition": "assistant-response and terminating-EOS tokens only",
        },
    )


def configure_engine() -> None:
    """Redirect the hardened masked-v2 engine into this isolated experiment."""
    validate_output_isolation()
    engine.BASE_CHECKPOINT = BASE_CHECKPOINT
    engine.BASE_GLOBAL_STEP = BASE_GLOBAL_STEP
    engine.TOKENIZER_FILE = TOKENIZER_FILE
    engine.DATA_FILE = DATA_FILE
    engine.MASK_FILE = MASK_FILE
    engine.DATASET_METADATA_FILE = METADATA_FILE
    engine.CHECKPOINT_DIR = CHECKPOINT_DIR
    engine.MAIN_CHECKPOINT = MAIN_CHECKPOINT
    engine.BEST_CHECKPOINT = BEST_CHECKPOINT
    engine.FORMAT_VERSION = FORMAT_VERSION
    engine.LEARNING_RATE = LEARNING_RATE
    engine.WEIGHT_DECAY = WEIGHT_DECAY
    engine.EPOCHS = EPOCHS
    engine.BATCH_SIZE = BATCH_SIZE
    engine.GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS
    engine.GRAD_CLIP = GRAD_CLIP
    engine.USE_AMP = USE_AMP
    engine.SEED = SEED
    engine.SAVE_EVERY_STEPS = SAVE_EVERY_STEPS
    engine.write_checkpoint_sidecar = write_sidecar


def split_counts(records: int) -> tuple[int, int, int]:
    train_records = int(records * 0.95)
    validation_records = records - train_records
    optimizer_steps = math.ceil(
        (train_records // BATCH_SIZE) / GRADIENT_ACCUMULATION_STEPS
    )
    return train_records, validation_records, optimizer_steps


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


def output_snapshot() -> set[Path]:
    return set(CHECKPOINT_DIR.rglob("*")) if CHECKPOINT_DIR.exists() else set()


def run_dry_run() -> None:
    """Run one forward-only batch; never step or write optimizer/checkpoint state."""
    before = output_snapshot()
    metadata = load_metadata()
    records, supervised_tokens = validate_inputs(metadata)
    validate_base_checkpoint()
    engine.ensure_no_fineweb_training_process()
    train_records, validation_records, optimizer_steps = split_counts(records)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_vasu_60m_config()
    model = VASUModel(config)
    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(f"unexpected parameter count: {parameters:,}")

    optimizer = build_optimizer(model, build_train_config())
    dataset = PackedInstructionDataset(
        str(DATA_FILE), str(MASK_FILE), config.max_seq_len,
        start_record=0, end_record=train_records,
    )
    inputs, targets, mask = next(iter(DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
        num_workers=0,
    )))
    inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)
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
        raise RuntimeError("dry run unexpectedly changed optimizer state")
    if output_snapshot() != before:
        raise RuntimeError("dry run wrote checkpoint output")
    peak_mib = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if device.type == "cuda" else None
    )
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print(f"Base global step: {BASE_GLOBAL_STEP}")
    print(f"Parameters: {parameters:,}")
    print(f"Records: {records:,}")
    print(f"Train records: {train_records:,}")
    print(f"Validation records: {validation_records:,}")
    print(f"Dataset supervised tokens: {supervised_tokens:,}")
    print(f"Batch shapes: {tuple(inputs.shape)}, {tuple(mask.shape)}")
    print(f"Batch supervised tokens: {batch_supervised:,}")
    print(f"Forward loss: {loss.item():.6f}")
    print(f"Calculated optimizer steps: {optimizer_steps}")
    print(f"Calculated target global step: {BASE_GLOBAL_STEP + optimizer_steps}")
    print(f"Peak CUDA memory: {peak_mib if peak_mib is not None else 'N/A'} MiB")
    print("Checkpoint writes: none")
    print("Training started: False")


def run_training() -> None:
    metadata = load_metadata()
    validate_inputs(metadata)
    validate_base_checkpoint()
    configure_engine()
    engine.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate one forward-only batch without writing checkpoints",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="explicitly authorize the one-epoch training entry point",
    )
    args = parser.parse_args()
    if args.dry_run == args.train:
        parser.error("choose exactly one of --dry-run or --train")
    if args.dry_run:
        run_dry_run()
    else:
        run_training()


if __name__ == "__main__":
    main()
