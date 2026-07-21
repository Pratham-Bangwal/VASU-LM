from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from vasu.data.instruction_quality import (
    build_release,
    cross_split_leakage,
    deterministic_split,
    load_jsonl,
    make_review_decision,
    sha256_file,
    write_jsonl,
)
from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.tokenizer.tokenizer import VASUTokenizer


DEMO = Path("data/examples/instruct/vasu_instruction_quality_v1_demo.jsonl")


def records() -> list[dict]:
    return load_jsonl(DEMO)


def build_config(tmp_path: Path, selected: list[dict], statuses: dict[str, str] | None = None) -> dict:
    source = tmp_path / "source.jsonl"
    review = tmp_path / "review.jsonl"
    write_jsonl(source, selected)
    statuses = statuses or {record["example_id"]: "approved" for record in selected}
    decisions = [
        make_review_decision(record, statuses[record["example_id"]], "tester", "reviewed")
        for record in selected
        if statuses.get(record["example_id"], "unreviewed") != "unreviewed"
    ]
    write_jsonl(review, decisions)
    return {
        "training_authorized": False,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "review_path": str(review),
        "tokenizer_path": "assets/tokenizer.json",
        "tokenizer_sha256": sha256_file(Path("assets/tokenizer.json")),
        "validation_ratio": 0.05,
        "split_seed": 42,
        "near_duplicate_threshold": 0.85,
        "sequence_length": 256,
        "release_path": str(tmp_path / "release.jsonl"),
        "token_path": str(tmp_path / "tokens.bin"),
        "mask_path": str(tmp_path / "mask.bin"),
        "release_manifest_path": str(tmp_path / "manifest.json"),
    }


def test_unapproved_and_rejected_records_are_excluded(tmp_path: Path) -> None:
    selected = records()
    statuses = {record["example_id"]: "approved" for record in selected}
    statuses[selected[0]["example_id"]] = "unreviewed"
    statuses[selected[1]["example_id"]] = "rejected"
    config = build_config(tmp_path, selected, statuses)
    manifest = build_release(config)
    assert selected[0]["example_id"] not in manifest["approved_example_ids"]
    assert selected[1]["example_id"] not in manifest["approved_example_ids"]
    assert len(manifest["approved_example_ids"]) == len(selected) - 2


def test_approved_records_are_included_once(tmp_path: Path) -> None:
    config = build_config(tmp_path, records())
    manifest = build_release(config)
    assert len(manifest["approved_example_ids"]) == len(set(manifest["approved_example_ids"])) == 21


def test_split_assignment_is_deterministic_and_stratified() -> None:
    first = deterministic_split(records(), 0.05, 42)
    assert first == deterministic_split(records(), 0.05, 42)
    assert set(first.values()) == {"train", "validation"}
    assert sum(value == "validation" for value in first.values()) == 7


def test_cross_split_duplicate_leakage_fails() -> None:
    left = records()[0]
    right = deepcopy(left)
    right["example_id"] = "viq1_999999"
    assignments = {left["example_id"]: "train", right["example_id"]: "validation"}
    assert cross_split_leakage([left, right], assignments, 0.85)


def test_token_and_mask_accounting_and_special_masks(tmp_path: Path) -> None:
    manifest = build_release(build_config(tmp_path, records()))
    tokens = np.fromfile(tmp_path / "tokens.bin", dtype=np.uint16)
    mask = np.fromfile(tmp_path / "mask.bin", dtype=np.uint8)
    assert len(tokens) == len(mask) == manifest["total_tokens"]
    assert set(np.unique(mask)) <= {0, 1}
    assert np.all(mask[tokens == 0] == 0)
    assert np.any((tokens == 3) & (mask == 1))
    assert int(mask.sum()) == manifest["assistant_loss_tokens"]
    assert int(tokens.max()) < 32000


def test_prompt_response_eos_and_padding_masks(tmp_path: Path) -> None:
    selected = records()
    config = build_config(tmp_path, selected)
    manifest = build_release(config)
    first_train_id = next(
        record["example_id"]
        for record in sorted(selected, key=lambda row: row["example_id"])
        if manifest["split_assignments"][record["example_id"]] == "train"
    )
    first = next(record for record in selected if record["example_id"] == first_train_id)
    tokenizer = VASUTokenizer()
    tokenizer.load("assets/tokenizer.json")
    prompt_ids = tokenizer.encode(
        format_alpaca_prompt(first["instruction"], first["input"])
    )
    response_ids = tokenizer.encode(f" {first['response']}")
    tokens = np.fromfile(config["token_path"], dtype=np.uint16)
    mask = np.fromfile(config["mask_path"], dtype=np.uint8)
    assert mask[: len(prompt_ids)].tolist() == [0] * len(prompt_ids)
    response_start = len(prompt_ids)
    response_end = response_start + len(response_ids)
    assert mask[response_start:response_end].tolist() == [1] * len(response_ids)
    assert tokens[response_end] == 3
    assert mask[response_end] == 1
    first_record_end = 257
    first_record_tokens = tokens[:first_record_end]
    final_nonpadding = int(np.flatnonzero(first_record_tokens != 0)[-1])
    assert np.all(mask[final_nonpadding + 1:first_record_end] == 0)


def test_train_and_validation_are_packed_separately(tmp_path: Path) -> None:
    manifest = build_release(build_config(tmp_path, records()))
    assert manifest["train_packed_records"] > 0
    assert manifest["validation_packed_records"] > 0
    assert manifest["records"] == manifest["train_packed_records"] + manifest["validation_packed_records"]


def test_deterministic_rebuild_is_byte_identical(tmp_path: Path) -> None:
    config = build_config(tmp_path, records())
    first = build_release(config)
    first_bytes = {name: Path(config[name]).read_bytes() for name in ("release_path", "token_path", "mask_path", "release_manifest_path")}
    second = build_release(config)
    assert first == second
    assert all(Path(config[name]).read_bytes() == value for name, value in first_bytes.items())


def test_source_hash_mismatch_fails(tmp_path: Path) -> None:
    config = build_config(tmp_path, records())
    config["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        build_release(config)


def test_tokenizer_hash_mismatch_fails(tmp_path: Path) -> None:
    config = build_config(tmp_path, records())
    config["tokenizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tokenizer SHA-256 mismatch"):
        build_release(config)


def test_training_authorization_must_remain_false(tmp_path: Path) -> None:
    config = build_config(tmp_path, records())
    config["training_authorized"] = True
    with pytest.raises(ValueError, match="training_authorized"):
        build_release(config)


def test_truncated_approved_response_is_rejected(tmp_path: Path) -> None:
    selected = records()
    selected[0] = deepcopy(selected[0])
    selected[0]["response"] = " ".join(f"distinct{index}" for index in range(600)) + "."
    config = build_config(tmp_path, selected)
    with pytest.raises(ValueError, match="truncated responses"):
        build_release(config)


def test_no_optimizer_or_training_code_in_release_module() -> None:
    source = Path("vasu/data/instruction_quality.py").read_text(encoding="utf-8")
    assert "optimizer.step" not in source
    assert "backward(" not in source


def test_manifest_is_hash_bound(tmp_path: Path) -> None:
    config = build_config(tmp_path, records())
    manifest = build_release(config)
    assert manifest["source_sha256"] == hashlib.sha256(Path(config["source_path"]).read_bytes()).hexdigest()
    assert manifest["tokenizer_sha256"] == sha256_file(Path("assets/tokenizer.json"))
    assert manifest["training_authorized"] is False
