from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vasu.training.dataset import ManifestTokenDataset, TextDataset


SEQ_LEN = 4


def write_tokens(path: Path, values: list[int]) -> Path:
    np.asarray(values, dtype=np.uint16).tofile(path)
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    tmp_path: Path,
    *,
    shard_values: list[list[int]] | None = None,
    validation_values: list[int] | None = None,
) -> tuple[Path, list[Path], Path]:
    shard_values = shard_values or [list(range(10)), list(range(100, 110))]
    validation_values = validation_values or list(range(200, 212))
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"test": true}', encoding="utf-8")

    shard_paths: list[Path] = []
    training_shards = []
    logical_start = 0
    for index, values in enumerate(shard_values):
        path = write_tokens(tmp_path / f"shard_{index}.bin", values)
        shard_paths.append(path)
        training_shards.append(
            {
                "path": str(path),
                "start_token": 0,
                "end_token": len(values),
                "role": f"train_{index}",
            }
        )
        logical_start += len(values)

    validation_path = write_tokens(tmp_path / "validation.bin", validation_values)
    manifest = {
        "format_version": 1,
        "dtype": "uint16",
        "tokenizer_path": str(tokenizer),
        "tokenizer_sha256": sha256(tokenizer),
        "sequence_length": SEQ_LEN,
        "training_shards": training_shards,
        "validation": {
            "path": str(validation_path),
            "start_token": 0,
            "end_token": len(validation_values),
            "role": "fixed_original_validation",
        },
        "logical_training_tokens": logical_start,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, shard_paths, validation_path


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_train_manifest_loading(tmp_path: Path):
    manifest, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(manifest, "train", SEQ_LEN)
    assert dataset.total_logical_tokens == 20
    assert len(dataset.shards) == 2


def test_valid_validation_manifest_loading(tmp_path: Path):
    manifest, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(manifest, "validation", SEQ_LEN)
    assert dataset.total_logical_tokens == 12
    assert dataset.shards[0].role == "fixed_original_validation"


def test_missing_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Manifest not found"):
        ManifestTokenDataset(tmp_path / "missing.json", "train", SEQ_LEN)


def test_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_unsupported_format_version(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    value = load_manifest(path)
    value["format_version"] = 2
    save_manifest(path, value)
    with pytest.raises(ValueError, match="Unsupported manifest"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_missing_shard(tmp_path: Path):
    path, shards, _ = build_manifest(tmp_path)
    shards[1].unlink()
    with pytest.raises(FileNotFoundError, match=r"training_shards\[1\]"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_invalid_physical_slice(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    value = load_manifest(path)
    value["training_shards"][0]["end_token"] = 999
    save_manifest(path, value)
    with pytest.raises(ValueError, match="Invalid physical slice"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_empty_training_shard(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    value = load_manifest(path)
    value["training_shards"][0]["end_token"] = 0
    save_manifest(path, value)
    with pytest.raises(ValueError, match="Invalid physical slice"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_invalid_dtype(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    value = load_manifest(path)
    value["dtype"] = "int32"
    save_manifest(path, value)
    with pytest.raises(ValueError, match="dtype"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_unknown_split_rejected(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        ManifestTokenDataset(path, "test", SEQ_LEN)


def test_logical_ranges_are_contiguous(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert dataset.shards[0].logical_end == dataset.shards[1].logical_start


@pytest.mark.parametrize(
    ("logical_start", "message"),
    [(9, "overlap"), (11, "gap")],
)
def test_logical_overlap_and_gap_rejected(
    tmp_path: Path, logical_start: int, message: str
):
    path, _, _ = build_manifest(tmp_path)
    value = load_manifest(path)
    value["training_shards"][1]["logical_start"] = logical_start
    value["training_shards"][1]["logical_end"] = logical_start + 10
    save_manifest(path, value)
    with pytest.raises(ValueError, match=message):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_single_shard_read(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert dataset._read_tokens(2, 5).tolist() == [2, 3, 4, 5, 6]


def test_cross_shard_read(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert dataset._read_tokens(8, 5).tolist() == [8, 9, 100, 101, 102]


def test_read_spanning_more_than_two_shards(tmp_path: Path):
    path, _, _ = build_manifest(
        tmp_path,
        shard_values=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    )
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert dataset._read_tokens(1, 7).tolist() == [2, 3, 4, 5, 6, 7, 8]


def test_exact_boundary_read(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert dataset._read_tokens(10, 4).tolist() == [100, 101, 102, 103]
    assert len(dataset.describe_read(10, 4)) == 1


def test_read_beyond_dataset_end(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    with pytest.raises(IndexError, match="exceeds"):
        dataset._read_tokens(18, 3)


def test_negative_logical_offset_and_nonpositive_count(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    with pytest.raises(ValueError, match="non-negative"):
        dataset._read_tokens(-1, 1)
    with pytest.raises(ValueError, match="positive"):
        dataset._read_tokens(0, 0)


def test_correct_dataset_length(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert len(dataset) == (20 - 1) // 4 == 4


def test_correct_target_shift_and_torch_long(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    x, y = dataset[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]
    assert x.dtype == torch.long and y.dtype == torch.long


def test_validation_never_reads_extension_data(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    validation = ManifestTokenDataset(path, "validation", SEQ_LEN)
    assert validation._read_tokens(0, 5).tolist() == [200, 201, 202, 203, 204]
    assert all(segment["role"] == "fixed_original_validation" for segment in validation.describe_read(0, 5))


@pytest.mark.parametrize(
    ("logical_start", "expected"),
    [(4, [4, 5, 6, 7]), (10, [100, 101, 102, 103]), (12, [102, 103, 104, 105])],
)
def test_requested_start_positions(
    tmp_path: Path, logical_start: int, expected: list[int]
):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(
        path,
        "train",
        SEQ_LEN,
        logical_start=logical_start,
        logical_end=logical_start + 5,
    )
    x, _ = dataset[0]
    assert x.tolist() == expected


def test_insufficient_tokens_rejected(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    with pytest.raises(ValueError, match="enough tokens"):
        ManifestTokenDataset(
            path, "train", SEQ_LEN, logical_start=16, logical_end=20
        )


def test_negative_index_rejected(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    with pytest.raises(IndexError, match="Negative"):
        dataset[-1]


def test_validation_overlap_rejected(tmp_path: Path):
    path, shards, _ = build_manifest(tmp_path)
    value = load_manifest(path)
    value["validation"] = {
        "path": str(shards[0]),
        "start_token": 5,
        "end_token": 10,
        "role": "fixed_original_validation",
    }
    save_manifest(path, value)
    with pytest.raises(ValueError, match="overlaps"):
        ManifestTokenDataset(path, "train", SEQ_LEN)


def test_existing_single_file_dataset_remains_compatible(tmp_path: Path):
    path = write_tokens(tmp_path / "legacy.bin", list(range(20)))
    dataset = TextDataset(data_file=str(path), seq_len=4, start=0, end=20)
    x, y = dataset[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]


def test_pickle_reopens_memmaps_for_worker_safety(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    assert dataset._read_tokens(0, 1).tolist() == [0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._memmaps == {}
    assert restored._read_tokens(8, 5).tolist() == [8, 9, 100, 101, 102]


def test_dataloader_current_worker_configuration(tmp_path: Path):
    path, _, _ = build_manifest(tmp_path)
    dataset = ManifestTokenDataset(path, "train", SEQ_LEN)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == y.shape == (2, SEQ_LEN)
    assert x[1].tolist() == [4, 5, 6, 7]
