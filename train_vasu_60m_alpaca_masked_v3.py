"""Isolated masked-Alpaca-v3 replication from the FineWeb step-200k base."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_vasu_60m_alpaca_masked_v2 as v2
from vasu.config import TrainConfig, get_vasu_60m_config
from vasu.data.alpaca_masked_v2 import (
    FORMAT_VERSION,
    atomic_write_json,
    validate_metadata_against_arrays,
)
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss
from vasu.training.optimizer import build_optimizer
from vasu.training.orchestration import (
    CheckpointInspection,
    inspect_checkpoint,
)


EXPERIMENT_NAME = "vasu_60m_alpaca_masked_v3_from_200k"
BASE_CHECKPOINT = Path(
    "checkpoints/vasu_60m/milestones/fineweb_step_200000.pt"
)
BASE_GLOBAL_STEP = 200_000
TOKENIZER_FILE = Path("assets/tokenizer.json")
DATA_FILE = Path("data/processed/instruct/alpaca_masked_v2.bin")
MASK_FILE = Path("data/processed/instruct/alpaca_masked_v2_mask.bin")
DATASET_METADATA_FILE = Path(
    "data/processed/instruct/alpaca_masked_v2_metadata.json"
)
CHECKPOINT_DIR = Path(
    "checkpoints/vasu_60m/alpaca_masked_v3_from_200k"
)
MAIN_CHECKPOINT = CHECKPOINT_DIR / "vasu.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

PROTECTED_DIRECTORIES = (
    Path("checkpoints/vasu_60m/alpaca"),
    Path("checkpoints/vasu_60m/alpaca_masked_v2"),
    Path("checkpoints/vasu_60m/fineweb_blocks"),
    Path("checkpoints/vasu_60m/milestones"),
)

# These aliases make the replication contract explicit and testable. They are
# intentionally identical to the proven v2 experiment.
EXPECTED_PARAMETERS = v2.EXPECTED_PARAMETERS
EPOCHS = v2.EPOCHS
BATCH_SIZE = v2.BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS = v2.GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE = v2.LEARNING_RATE
WEIGHT_DECAY = v2.WEIGHT_DECAY
GRAD_CLIP = v2.GRAD_CLIP
USE_AMP = v2.USE_AMP
SEED = v2.SEED
SAVE_EVERY_STEPS = v2.SAVE_EVERY_STEPS


@dataclass(frozen=True)
class DatasetValidation:
    records: int
    sequence_length: int
    total_tokens: int
    supervised_tokens: int
    tokenizer_vocab_size: int
    tokenizer_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_isolation(checkpoint_dir: Path = CHECKPOINT_DIR) -> None:
    """Reject any output location that overlaps a protected experiment."""

    output = checkpoint_dir.resolve()
    expected = CHECKPOINT_DIR.resolve()
    if output != expected:
        raise ValueError(f"v3 output must be exactly {CHECKPOINT_DIR}")
    for protected in PROTECTED_DIRECTORIES:
        protected_path = protected.resolve()
        if output == protected_path or protected_path in output.parents:
            raise ValueError(f"v3 output overlaps protected path: {protected}")


def load_dataset_metadata(
    metadata_path: Path = DATASET_METADATA_FILE,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != FORMAT_VERSION:
        raise ValueError("unexpected masked-Alpaca-v2 dataset format")
    if int(metadata.get("sequence_length", -1)) != 256:
        raise ValueError("masked-Alpaca-v2 sequence length must be 256")
    if int(metadata.get("record_length", -1)) != 257:
        raise ValueError("masked-Alpaca-v2 record length must be 257")
    return metadata


def validate_tokenizer(
    metadata: dict[str, Any],
    tokenizer_path: Path = TOKENIZER_FILE,
) -> tuple[VASUTokenizer, str]:
    if not tokenizer_path.is_file():
        raise FileNotFoundError(tokenizer_path)
    recorded_path = Path(str(metadata.get("tokenizer_path", "")))
    if recorded_path.as_posix().replace("\\", "/") != tokenizer_path.as_posix():
        raise ValueError(
            f"dataset tokenizer path mismatch: {recorded_path} != {tokenizer_path}"
        )
    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    actual_vocab = tokenizer.tokenizer.get_vocab_size()
    expected_vocab = int(metadata.get("tokenizer_vocab_size", -1))
    model_vocab = get_vasu_60m_config().vocab_size
    if actual_vocab != expected_vocab or actual_vocab != model_vocab:
        raise ValueError(
            "tokenizer vocabulary mismatch: "
            f"actual={actual_vocab}, metadata={expected_vocab}, "
            f"model={model_vocab}"
        )
    pad_id = tokenizer.tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    if pad_id != int(metadata["pad_token_id"]):
        raise ValueError("PAD token ID does not match dataset metadata")
    if eos_id != int(metadata["eos_token_id"]):
        raise ValueError("EOS token ID does not match dataset metadata")
    return tokenizer, _sha256(tokenizer_path)


def validate_dataset(
    metadata: dict[str, Any],
    tokenizer_sha256: str,
    data_path: Path = DATA_FILE,
    mask_path: Path = MASK_FILE,
) -> DatasetValidation:
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    if len(tokens) != len(mask):
        raise ValueError("token and mask files have different lengths")
    record_length = int(metadata["record_length"])
    if len(tokens) % record_length:
        raise ValueError("packed files are not divisible into fixed records")
    token_records = tokens.reshape(-1, record_length)
    mask_records = mask.reshape(-1, record_length)
    validate_metadata_against_arrays(token_records, mask_records, metadata)

    # Stored masks describe current tokens; mask[:, 1:] is the exact mask for
    # y=tokens[:, 1:]. Prompt/PAD targets remain zero and response/EOS targets
    # remain one, matching PackedInstructionDataset and v2.
    target_mask = mask_records[:, 1:]
    if not np.any(target_mask == 1):
        raise ValueError("dataset contains no supervised assistant targets")
    eos_id = int(metadata["eos_token_id"])
    supervised_eos = (token_records == eos_id) & (mask_records == 1)
    if int(supervised_eos.sum()) != int(metadata["eos_tokens"]):
        raise ValueError("assistant EOS supervision does not match metadata")
    return DatasetValidation(
        records=int(token_records.shape[0]),
        sequence_length=int(metadata["sequence_length"]),
        total_tokens=int(token_records.size),
        supervised_tokens=int(mask_records.sum()),
        tokenizer_vocab_size=int(metadata["tokenizer_vocab_size"]),
        tokenizer_sha256=tokenizer_sha256,
    )


def validate_base_checkpoint(
    checkpoint_path: Path = BASE_CHECKPOINT,
) -> CheckpointInspection:
    inspection = inspect_checkpoint(
        checkpoint_path,
        verify_finite=True,
        calculate_hash=True,
    )
    if not inspection.valid:
        raise RuntimeError(f"invalid FineWeb base checkpoint: {inspection.error}")
    if inspection.global_step != BASE_GLOBAL_STEP:
        raise RuntimeError(
            f"base checkpoint global_step must be {BASE_GLOBAL_STEP}; "
            f"found {inspection.global_step}"
        )
    if inspection.model_config != "vasu_60m":
        raise RuntimeError("base checkpoint is not a VASU-60M checkpoint")
    if inspection.finite_tensors is not True:
        raise RuntimeError("base checkpoint finite-tensor validation failed")
    return inspection


def configure_v2_replica() -> None:
    """Redirect the proven v2 engine exclusively into the v3 experiment."""

    validate_output_isolation()
    v2.BASE_CHECKPOINT = BASE_CHECKPOINT
    v2.BASE_GLOBAL_STEP = BASE_GLOBAL_STEP
    v2.TOKENIZER_FILE = TOKENIZER_FILE
    v2.DATA_FILE = DATA_FILE
    v2.MASK_FILE = MASK_FILE
    v2.DATASET_METADATA_FILE = DATASET_METADATA_FILE
    v2.CHECKPOINT_DIR = CHECKPOINT_DIR
    v2.MAIN_CHECKPOINT = MAIN_CHECKPOINT
    v2.BEST_CHECKPOINT = BEST_CHECKPOINT
    v2.write_checkpoint_sidecar = write_v3_checkpoint_sidecar


def write_v3_checkpoint_sidecar(
    checkpoint_path: Path,
    global_step: int,
    supervised_tokens_processed: int,
) -> None:
    """Record v3 provenance without modifying the checkpoint payload."""

    if CHECKPOINT_DIR.resolve() not in checkpoint_path.resolve().parents:
        raise ValueError("refusing to write a v3 sidecar outside v3 output")
    atomic_write_json(
        checkpoint_path.with_suffix(".metadata.json"),
        {
            "experiment": EXPERIMENT_NAME,
            "checkpoint": str(checkpoint_path),
            "global_step": global_step,
            "base_checkpoint": str(BASE_CHECKPOINT),
            "base_global_step": BASE_GLOBAL_STEP,
            "dataset_format_version": FORMAT_VERSION,
            "dataset_metadata": str(DATASET_METADATA_FILE),
            "supervised_tokens_processed": supervised_tokens_processed,
            "loss_definition": (
                "assistant-response and terminating-EOS tokens only"
            ),
        },
    )


def find_v3_resume() -> tuple[Path, dict] | None:
    configure_v2_replica()
    return v2.find_latest_checkpoint()


def initialization_label(resume: tuple[Path, dict] | None) -> str:
    if resume is None:
        return "Initialization source: FineWeb step-200000 base"
    return "Resume source: Alpaca masked v3 checkpoint"


def strict_load_model_state(
    model: torch.nn.Module,
    model_state: dict[str, Any],
) -> None:
    model.load_state_dict(model_state, strict=True)


def output_snapshot(checkpoint_dir: Path = CHECKPOINT_DIR) -> set[Path]:
    if not checkpoint_dir.exists():
        return set()
    return set(checkpoint_dir.rglob("*"))


def _build_train_config() -> TrainConfig:
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


def run_dry_run() -> None:
    """Validate the complete fresh path without stepping or writing state."""

    validate_output_isolation()
    before_files = output_snapshot()
    metadata = load_dataset_metadata()
    tokenizer, tokenizer_hash = validate_tokenizer(metadata)
    dataset_summary = validate_dataset(metadata, tokenizer_hash)
    base = validate_base_checkpoint()
    v2.ensure_no_fineweb_training_process()

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = get_vasu_60m_config()
    model = VASUModel(model_config)
    checkpoint = torch.load(
        BASE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    strict_load_model_state(model, checkpoint["model"])
    del checkpoint
    model.to(device)
    if v2.count_parameters(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("unexpected VASU-60M parameter count")

    train_config = _build_train_config()
    optimizer = build_optimizer(model, train_config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    split_record = int(dataset_summary.records * 0.95)
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
    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )
    inputs, targets, mask = next(iter(loader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    supervised_tokens = int(mask.sum().item())
    if supervised_tokens <= 0:
        raise RuntimeError("dry-run batch has zero supervised tokens")
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda" and USE_AMP):
            logits = model(inputs)
            loss = language_model_loss(logits, targets, mask)
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run masked loss is not finite")

    # Constructing these objects is part of the dry-run contract. No scheduler
    # or optimizer step is allowed.
    if optimizer.state:
        raise RuntimeError("dry run unexpectedly populated optimizer state")
    if scheduler.last_epoch != 0:
        raise RuntimeError("dry run unexpectedly advanced the scheduler")
    after_files = output_snapshot()
    if after_files != before_files:
        raise RuntimeError("dry run wrote into the v3 checkpoint directory")

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Base checkpoint: {BASE_CHECKPOINT}")
    print(f"Base global step: {base.global_step}")
    print("Model configuration: VASU-60M")
    print(f"Parameter count: {v2.count_parameters(model):,}")
    print(f"Tokenizer: {TOKENIZER_FILE} (SHA-256 {tokenizer_hash})")
    print(f"Dataset: {DATA_FILE}")
    print(f"Mask: {MASK_FILE}")
    print(f"Train records: {len(train_dataset):,}")
    print(f"Validation records: {len(validation_dataset):,}")
    print(f"Sequence length: {model_config.max_seq_len}")
    print(f"Input shape: {tuple(inputs.shape)}")
    print(f"Target shape: {tuple(targets.shape)}")
    print(f"Mask shape: {tuple(mask.shape)}")
    print(f"Supervised tokens in test batch: {supervised_tokens}")
    print(f"Forward loss: {loss.item():.6f}")
    print(f"Output directory: {CHECKPOINT_DIR}")
    print("Training started: False")
    _ = tokenizer  # Keep tokenizer validation explicit through report creation.


def run_training() -> None:
    metadata = load_dataset_metadata()
    _, tokenizer_hash = validate_tokenizer(metadata)
    validate_dataset(metadata, tokenizer_hash)
    validate_base_checkpoint()
    configure_v2_replica()
    resume = v2.find_latest_checkpoint()
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(initialization_label(resume))
    if resume is not None:
        del resume
    v2.main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and run one forward-only batch without writing state",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_run()
    else:
        run_training()


if __name__ == "__main__":
    main()
