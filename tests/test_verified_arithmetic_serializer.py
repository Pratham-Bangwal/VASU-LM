from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.serialize_vasu_verified_arithmetic_v1 as serializer
from scripts.serialize_vasu_verified_arithmetic_v1 import (
    RECORD_WIDTH,
    serialize_release,
    validate_release,
)
from vasu.training.instruction_dataset import PackedInstructionDataset


TOKENIZER = Path("assets/tokenizer.json")
COUNTS = {"train": 28, "development": 7, "evaluation": 7}
CREATED_AT = "2026-07-24T00:00:00+00:00"


def _serialize(path: Path, **kwargs: object) -> dict[str, object]:
    return serialize_release(
        tokenizer_path=TOKENIZER,
        output_dir=path,
        counts=COUNTS,
        seed=42,
        created_at=CREATED_AT,
        **kwargs,
    )


def test_release_is_deterministic_complete_and_hash_bound(tmp_path: Path) -> None:
    first = _serialize(tmp_path / "first")
    second = _serialize(tmp_path / "second")

    assert first == second
    assert first["dataset_id"] == "verified_arithmetic_v1"
    assert first["training_authorized"] is False
    assert first["tokenizer"]["sha256"] == (
        "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
    )
    assert first["tokenizer"]["special_token_ids"] == {
        "bos": 2,
        "eos": 3,
        "pad": 0,
    }
    assert first["logical_example_counts"] == COUNTS
    assert first["packing"]["unique_examples_consumed"] == COUNTS["train"]
    assert first["packing"]["total_logical_examples_consumed"] == COUNTS["train"]
    assert first["packing"]["replay_epochs"] == 1
    assert first["validation"] == {
        "all_answers_verified": True,
        "capability_v1_exclusion_check": True,
        "pad_absent_from_logical_examples": True,
        "split_overlap_check": True,
        "terminal_eos_check": True,
        "tokenizer_round_trip_check": True,
    }
    for artifact in first["artifacts"]:
        assert (
            (tmp_path / "first" / artifact).read_bytes()
            == (tmp_path / "second" / artifact).read_bytes()
        )
        assert not Path(first["artifacts"][artifact]["path"]).is_absolute()
    assert not Path(first["tokenizer"]["path"]).is_absolute()
    assert not Path(first["generator"]["path"]).is_absolute()
    assert not Path(first["serializer"]["path"]).is_absolute()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_logical_eval_artifacts_keep_exact_answers_and_fields(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    manifest = _serialize(output)
    for name, split in (("dev.jsonl", "development"), ("eval.jsonl", "evaluation")):
        records = [
            json.loads(line)
            for line in (output / name).read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == COUNTS[split]
        assert all(record["split"] == split for record in records)
        assert all(record["prompt"] and record["answer"] for record in records)
        assert all(record["operand_metadata"] for record in records)
        assert all(record["template_id"] == "question_answer_v1" for record in records)
        assert all("[EOS]" not in record["text"] for record in records)
    assert manifest["logical_split_sha256"]["development"]
    assert manifest["logical_split_sha256"]["evaluation"]


def test_binary_shapes_masks_and_packed_dataset_compatibility(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    manifest = _serialize(output)
    record_count = manifest["packing"]["packed_records"]
    tokens = np.fromfile(output / "train_tokens.bin", dtype=np.uint16).reshape(
        record_count, RECORD_WIDTH
    )
    masks = np.fromfile(output / "train_loss_mask.bin", dtype=np.uint8).reshape(
        record_count, RECORD_WIDTH
    )
    pad_id = manifest["tokenizer"]["special_token_ids"]["pad"]
    eos_id = manifest["tokenizer"]["special_token_ids"]["eos"]
    assert np.all(masks[:, 0] == 0)
    assert np.all(masks[:, 1:][tokens[:, 1:] == pad_id] == 0)
    assert np.all(masks[:, 1:][tokens[:, :-1] == pad_id] == 0)
    assert np.any(tokens == eos_id)
    dataset = PackedInstructionDataset(
        str(output / "train_tokens.bin"),
        str(output / "train_loss_mask.bin"),
        256,
    )
    x, y, loss_mask = dataset[0]
    assert x.shape == y.shape == loss_mask.shape == torch.Size([256])
    assert torch.equal(loss_mask, torch.tensor(masks[0, 1:], dtype=torch.float32))


def test_validation_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "release"
    _serialize(output)
    with (output / "train_tokens.bin").open("ab") as handle:
        handle.write(b"\x00\x00")
    with pytest.raises(ValueError, match="byte-size mismatch"):
        validate_release(output)


def test_validation_rejects_mask_tokenizer_and_manifest_corruption(
    tmp_path: Path,
) -> None:
    mask_output = tmp_path / "mask-release"
    _serialize(mask_output)
    mask = mask_output / "train_loss_mask.bin"
    content = bytearray(mask.read_bytes())
    content[-1] ^= 1
    mask.write_bytes(content)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_release(mask_output)

    tokenizer_output = tmp_path / "tokenizer-release"
    _serialize(tokenizer_output)
    wrong_tokenizer = tmp_path / "wrong-tokenizer.json"
    wrong_tokenizer.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        validate_release(tokenizer_output, tokenizer_path=wrong_tokenizer)

    manifest_output = tmp_path / "manifest-release"
    _serialize(manifest_output)
    (manifest_output / "manifest.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        validate_release(manifest_output)


def test_overwrite_is_explicit_and_failed_build_preserves_release(
    tmp_path: Path,
) -> None:
    output = tmp_path / "release"
    original = _serialize(output)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _serialize(output)

    missing_tokenizer = tmp_path / "missing-tokenizer.json"
    with pytest.raises(FileNotFoundError):
        serialize_release(
            tokenizer_path=missing_tokenizer,
            output_dir=output,
            counts=COUNTS,
            seed=42,
            overwrite=True,
            created_at=CREATED_AT,
        )
    assert validate_release(output) == original

    replaced = _serialize(output, overwrite=True)
    assert validate_release(output) == replaced


def test_dry_run_writes_no_release_or_orphan_staging(tmp_path: Path) -> None:
    output = tmp_path / "dry-run-release"
    manifest = _serialize(output, dry_run=True)
    assert manifest["packing"]["unique_examples_consumed"] == COUNTS["train"]
    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_failed_staged_write_cleans_up_and_preserves_previous_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "release"
    original = _serialize(output)

    def fail_manifest(*args: object, **kwargs: object) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(serializer, "_write_json", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        _serialize(output, overwrite=True)
    monkeypatch.undo()
    assert validate_release(output) == original
    assert not list(tmp_path.glob(".*.staging-*"))
