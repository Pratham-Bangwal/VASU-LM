"""Validate the production FineWeb manifest and optional model dry runs.

This script never trains beyond three isolated forward/backward checks and never
saves a checkpoint. Model execution requires the explicit ``--dry-run`` flag.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from vasu.config import get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.training.dataset import ManifestTokenDataset
from vasu.training.losses import language_model_loss


DEFAULT_MANIFEST = Path("data/processed/pretrain/fineweb_manifest.json")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/vasu_60m/milestones/fineweb_step_150000.pt"
)
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
SEQUENCE_LENGTH = 256
START_STEP = 150_000
TARGET_STEP = 200_000
ORIGINAL_BOUNDARY = 1_245_891_252
EXPECTED_EXTENSION_OFFSET_AT_TARGET = 392_508_748
EXPECTED_TOTAL_TOKENS = 1_745_891_730
EXPECTED_REMAINING_TOKENS = 107_491_730


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Opt in to three isolated VASU-60M forward/backward checks.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manual_slice(path: Path, start: int, end: int) -> np.ndarray:
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    return np.asarray(tokens[start:end])


def validate_mappings(manifest_path: Path) -> ManifestTokenDataset:
    dataset = ManifestTokenDataset(
        manifest_path=manifest_path,
        split="train",
        sequence_length=SEQUENCE_LENGTH,
    )
    validation = ManifestTokenDataset(
        manifest_path=manifest_path,
        split="validation",
        sequence_length=SEQUENCE_LENGTH,
    )
    assert dataset.total_logical_tokens == EXPECTED_TOTAL_TOKENS

    first = dataset.describe_read(0, 1)[0]
    last_original = dataset.describe_read(ORIGINAL_BOUNDARY - 1, 1)[0]
    first_extension = dataset.describe_read(ORIGINAL_BOUNDARY, 1)[0]
    assert first["role"] == "original_train" and first["physical_start"] == 0
    assert last_original["physical_start"] == ORIGINAL_BOUNDARY - 1
    assert first_extension["role"] == "extension_train"
    assert first_extension["physical_start"] == 0

    step_150000_offset = (
        START_STEP * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQUENCE_LENGTH
    )
    step_200000_offset = (
        TARGET_STEP * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQUENCE_LENGTH
    )
    assert step_150000_offset == 1_228_800_000
    assert step_200000_offset == 1_638_400_000
    target_location = dataset.describe_read(step_200000_offset, 1)[0]
    assert target_location["role"] == "extension_train"
    assert target_location["physical_start"] == EXPECTED_EXTENSION_OFFSET_AT_TARGET
    assert dataset.total_logical_tokens - step_200000_offset == EXPECTED_REMAINING_TOKENS

    validation_shard = validation.shards[0]
    assert validation_shard.role == "fixed_original_validation"
    assert validation_shard.physical_start == ORIGINAL_BOUNDARY
    assert validation_shard.physical_end == 1_271_317_605

    original_path = dataset.shards[0].path
    extension_path = dataset.shards[1].path

    cross_start = ORIGINAL_BOUNDARY - 100
    cross_tokens = np.asarray(dataset._read_tokens(cross_start, 257))
    expected_cross = np.concatenate(
        (
            manual_slice(original_path, cross_start, ORIGINAL_BOUNDARY),
            manual_slice(extension_path, 0, 157),
        )
    )
    assert np.array_equal(cross_tokens, expected_cross)
    composition = dataset.describe_read(cross_start, 257)
    assert [segment["token_count"] for segment in composition] == [100, 157]

    read_cases = {
        "entirely_before_boundary": ORIGINAL_BOUNDARY - 1_000,
        "exactly_at_boundary": ORIGINAL_BOUNDARY,
        "inside_extension": ORIGINAL_BOUNDARY + 10_000,
        "final_valid_read": dataset.total_logical_tokens - 257,
    }
    for name, logical_start in read_cases.items():
        tokens = dataset._read_tokens(logical_start, 257)
        assert len(tokens) == 257, name

    print("Production manifest validation: PASS")
    print("Logical offset 0 -> original token 0")
    print(
        f"Logical offset {ORIGINAL_BOUNDARY - 1:,} -> final original training token"
    )
    print(f"Logical offset {ORIGINAL_BOUNDARY:,} -> extension token 0")
    print(f"Step 150000 logical offset: {step_150000_offset:,}")
    print(f"Step 200000 logical offset: {step_200000_offset:,}")
    print(
        "Step 200000 extension offset: "
        f"{target_location['physical_start']:,}"
    )
    print(f"Available logical training tokens: {dataset.total_logical_tokens:,}")
    print(f"Remaining tokens after target: {EXPECTED_REMAINING_TOKENS:,}")
    print("Capacity status: PASS")
    print("Cross-boundary composition: 100 original + 157 extension tokens")
    print("Validation isolation: PASS")
    return dataset


def run_location_dry_test(
    *,
    name: str,
    logical_start: int,
    dataset: ManifestTokenDataset,
    model: VASUModel,
    device: torch.device,
) -> float:
    local_dataset = ManifestTokenDataset(
        manifest_path=dataset.manifest_path,
        split="train",
        sequence_length=SEQUENCE_LENGTH,
        logical_start=logical_start,
        logical_end=logical_start + BATCH_SIZE * SEQUENCE_LENGTH + 1,
    )
    inputs = torch.stack([local_dataset[index][0] for index in range(BATCH_SIZE)])
    targets = torch.stack([local_dataset[index][1] for index in range(BATCH_SIZE)])
    composition = dataset.describe_read(
        logical_start, BATCH_SIZE * SEQUENCE_LENGTH + 1
    )

    model.train()
    model.zero_grad(set_to_none=True)
    inputs = inputs.to(device)
    targets = targets.to(device)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        logits = model(inputs)
        loss = language_model_loss(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError(f"{name}: non-finite loss {loss.item()}")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"{name}: missing or non-finite gradients")

    print(f"\n{name} dry run: PASS")
    print(f"  logical_start: {logical_start:,}")
    print(f"  input shape: {tuple(inputs.shape)}")
    print(f"  target shape: {tuple(targets.shape)}")
    print(f"  shard composition: {composition}")
    print(f"  finite loss: {loss.item():.6f}")
    print("  finite gradients: True")
    return float(loss.item())


def run_model_dry_tests(
    dataset: ManifestTokenDataset,
    checkpoint_path: Path,
) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Dry-run checkpoint not found: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = VASUModel(get_vasu_60m_config()).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if int(checkpoint.get("global_step", -1)) != START_STEP:
        raise ValueError(
            f"Expected checkpoint global_step {START_STEP}, found "
            f"{checkpoint.get('global_step')!r}."
        )
    model.load_state_dict(checkpoint["model"])
    print(f"\nLoaded dry-run checkpoint: {checkpoint_path}")
    print(f"Checkpoint global_step: {checkpoint['global_step']}")
    print(f"Dry-run device: {device}")

    locations = (
        ("Original-shard", 1_228_800_000),
        ("Boundary-crossing", ORIGINAL_BOUNDARY - 100),
        ("Extension-shard", ORIGINAL_BOUNDARY + 10_000),
    )
    for name, logical_start in locations:
        run_location_dry_test(
            name=name,
            logical_start=logical_start,
            dataset=dataset,
            model=model,
            device=device,
        )

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\nPeak CUDA memory: {peak:.2f} MiB")
    else:
        print("\nPeak CUDA memory: unavailable (CPU dry run)")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    dataset = validate_mappings(manifest_path)

    protected_paths = (
        dataset.shards[0].path,
        dataset.shards[1].path,
        manifest_path,
        dataset.tokenizer_path,
        checkpoint_path,
    )
    before = {path: sha256_file(path) for path in protected_paths}

    step_150000_offset = (
        START_STEP * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQUENCE_LENGTH
    )
    simulated_after_boundary_step = 160_000
    simulated_after_boundary_offset = (
        simulated_after_boundary_step
        * BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
        * SEQUENCE_LENGTH
    )
    assert step_150000_offset == 1_228_800_000
    assert simulated_after_boundary_offset > ORIGINAL_BOUNDARY
    assert dataset.describe_read(step_150000_offset, 1)[0]["role"] == "original_train"
    assert (
        dataset.describe_read(simulated_after_boundary_offset, 1)[0]["role"]
        == "extension_train"
    )
    boundary_sample_index = (ORIGINAL_BOUNDARY - step_150000_offset) // SEQUENCE_LENGTH
    boundary_sample_start = (
        step_150000_offset + boundary_sample_index * SEQUENCE_LENGTH
    )
    next_sample_start = boundary_sample_start + SEQUENCE_LENGTH
    boundary_sample_parts = dataset.describe_read(
        boundary_sample_start, SEQUENCE_LENGTH + 1
    )
    assert len(boundary_sample_parts) == 2
    assert boundary_sample_parts[0]["role"] == "original_train"
    assert boundary_sample_parts[1]["role"] == "extension_train"
    assert dataset.describe_read(next_sample_start, 1)[0]["role"] == "extension_train"
    print("Resume-position dry test: PASS")
    print(f"  step 150000 -> {step_150000_offset:,}")
    print(
        f"  simulated step {simulated_after_boundary_step} -> "
        f"{simulated_after_boundary_offset:,} (extension)"
    )
    print(
        f"  sample {boundary_sample_index:,} from step-150000 start crosses "
        f"the boundary at logical offset {boundary_sample_start:,}"
    )
    print("  sample stride: 256; replay/skip/validation leakage: none")

    if args.dry_run:
        run_model_dry_tests(dataset, checkpoint_path)
    else:
        print("Model dry runs skipped; pass --dry-run to opt in.")

    after = {path: sha256_file(path) for path in protected_paths}
    if before != after:
        changed = [str(path) for path in protected_paths if before[path] != after[path]]
        raise RuntimeError(f"Protected files changed during validation: {changed}")
    print("Protected file hashes unchanged: PASS")
    print("Full continuation training started: False")


if __name__ == "__main__":
    main()
