"""Determinism, verification, isolation, and serialization tests for v2."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.serialize_vasu_verified_arithmetic_v2 import (
    serialize_release,
    validate_release,
)
from vasu.data.arithmetic_v2 import (
    CATEGORIES,
    FORBIDDEN_PROMPT_FRAGMENTS,
    generate_records,
    recompute_answer,
)
from vasu.training.instruction_dataset import PackedInstructionDataset


TOKENIZER = Path("assets/tokenizer.json")
COUNTS = {"train": 880, "development": 110, "evaluation": 110}


def _release(path: Path) -> dict[str, object]:
    return serialize_release(
        tokenizer_path=TOKENIZER,
        output_dir=path,
        counts=COUNTS,
        seed=42,
        created_at="2026-07-24T00:00:00+00:00",
        minimum_packed_records=1,
    )


def test_v2_generator_is_deterministic_stable_and_exact() -> None:
    first = generate_records("train", 2_200, 42)
    second = generate_records("train", 2_200, 42)
    assert first == second
    assert len({record["id"] for record in first}) == len(first)
    assert len({record["normalized_expression_sha256"] for record in first}) == len(first)
    assert set(record["operation"] for record in first) == set(CATEGORIES)
    assert all(recompute_answer(record) == record["answer"] for record in first)
    assert all(
        fragment not in record["prompt"]
        for record in first
        for fragment in FORBIDDEN_PROMPT_FRAGMENTS
    )


def test_v2_splits_have_operand_expression_and_template_holdouts() -> None:
    splits = {
        split: generate_records(split, 330, 42)
        for split in ("train", "development", "evaluation")
    }
    expression_sets = [
        {record["normalized_expression_sha256"] for record in records}
        for records in splits.values()
    ]
    assert not expression_sets[0] & expression_sets[1]
    assert not expression_sets[0] & expression_sets[2]
    assert not expression_sets[1] & expression_sets[2]
    train_templates = {record["template_id"] for record in splits["train"]}
    heldout = {
        record["template_id"]
        for split in ("development", "evaluation")
        for record in splits[split]
    }
    assert not train_templates & heldout


def test_fraction_signed_division_and_normalization_are_exact() -> None:
    records = generate_records("train", 4_400, 17)
    fractions = [record for record in records if record["operation"] == "fraction"]
    divisions = [record for record in records if record["operation"] == "exact_division"]
    signed = [
        record
        for record in records
        if record["operand_metadata"]["sign_pattern"] == "signed"
    ]
    assert fractions and divisions and signed
    assert all(recompute_answer(record) == record["answer"] for record in fractions + divisions + signed)
    for record in fractions:
        answer = record["answer"]
        if "/" in answer:
            numerator, denominator = map(int, answer.split("/"))
            assert Fraction(numerator, denominator).denominator == denominator


def test_v2_release_is_deterministic_hash_bound_and_mask_compatible(
    tmp_path: Path,
) -> None:
    first = _release(tmp_path / "first")
    second = _release(tmp_path / "second")
    assert first == second
    assert first["dataset_id"] == "verified_arithmetic_v2"
    assert first["training_authorized"] is False
    assert first["packing"]["replay_epochs"] == 1
    assert first["packing"]["unique_examples_consumed"] == COUNTS["train"]
    for name, metadata in first["artifacts"].items():
        assert metadata["sha256"] == second["artifacts"][name]["sha256"]
        assert (
            (tmp_path / "first" / name).read_bytes()
            == (tmp_path / "second" / name).read_bytes()
        )
    dataset = PackedInstructionDataset(
        str(tmp_path / "first" / "train_tokens.bin"),
        str(tmp_path / "first" / "train_loss_mask.bin"),
        256,
    )
    x, y, mask = dataset[0]
    assert x.shape == y.shape == mask.shape == torch.Size([256])
    tokens = np.fromfile(
        tmp_path / "first" / "train_tokens.bin", dtype=np.uint16
    ).reshape(-1, 257)
    masks = np.fromfile(
        tmp_path / "first" / "train_loss_mask.bin", dtype=np.uint8
    ).reshape(-1, 257)
    assert np.all(masks[:, 1:][tokens[:, 1:] == 0] == 0)


def test_v2_validation_detects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "release"
    _release(output)
    content = bytearray((output / "train_loss_mask.bin").read_bytes())
    content[-1] ^= 1
    (output / "train_loss_mask.bin").write_bytes(content)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_release(output)


def test_v2_logical_eval_remains_directly_accessible(tmp_path: Path) -> None:
    output = tmp_path / "release"
    manifest = _release(output)
    records = [
        json.loads(line)
        for line in (output / "eval.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == COUNTS["evaluation"]
    assert all(record["prompt"] and record["answer"] for record in records)
    assert manifest["logical_example_counts"]["evaluation"] == len(records)
