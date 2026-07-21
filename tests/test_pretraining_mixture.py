from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import train_vasu_60m_factual_cpt as training
from vasu.data.mixtures.schemas import MixtureManifest, MixtureSource
from vasu.data.pretraining_mixture import (
    FixedRecordTokenDataset,
    allocate_sequences,
    build_mixture_artifact,
    build_source_schedule,
    deterministic_parent_split,
    prepare_wikimedia_tokenized_split,
)
from vasu.training.dataset import TextDataset


TOKENIZER = Path("assets/tokenizer.json").resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(parent: str, chunk: int) -> dict[str, object]:
    sentence = f"A factual paragraph for parent {parent} and chunk {chunk}."
    return {
        "parent_document_id": parent,
        "chunk_id": f"{parent}:{chunk:04d}",
        "cleaned_text": " ".join([sentence] * 4),
        "token_count": 48,
        "provenance_metadata": {"source_row_index": int(parent)},
    }


def _prepare_small_wiki(tmp_path: Path) -> tuple[dict, Path, Path, Path, Path]:
    source = tmp_path / "source.jsonl"
    records = [_source_record(str(parent), chunk) for parent in range(1, 21) for chunk in range(2)]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    train = tmp_path / "wiki_train.bin"
    validation = tmp_path / "wiki_validation.bin"
    manifest = tmp_path / "wiki_manifest.json"
    result = prepare_wikimedia_tokenized_split(
        source_path=source,
        expected_source_sha256=_sha(source),
        tokenizer_path=TOKENIZER,
        expected_tokenizer_sha256=_sha(TOKENIZER),
        output_train_path=train,
        output_validation_path=validation,
        output_manifest_path=manifest,
        seed=42,
    )
    return result, source, train, validation, manifest


def _mixture_config(tmp_path: Path, seed: int = 42, budget: int = 5120) -> Path:
    wiki_meta, _, wiki_train, _, wiki_manifest = _prepare_small_wiki(tmp_path)
    fine_original = tmp_path / "fine_original.bin"
    fine_extension = tmp_path / "fine_extension.bin"
    np.arange(20_000, dtype=np.uint16).tofile(fine_original)
    np.arange(20_000, 40_000, dtype=np.uint16).tofile(fine_extension)
    fine_manifest = tmp_path / "fine_manifest.json"
    fine_manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "dtype": "uint16",
                "tokenizer_path": str(TOKENIZER),
                "tokenizer_sha256": _sha(TOKENIZER),
                "sequence_length": 256,
                "training_shards": [
                    {
                        "path": str(fine_original), "start_token": 0,
                        "end_token": 19000, "role": "original"
                    },
                    {
                        "path": str(fine_extension), "start_token": 0,
                        "end_token": 20000, "role": "extension"
                    },
                ],
                "validation": {
                    "path": str(fine_original), "start_token": 19000,
                    "end_token": 20000, "role": "validation"
                },
                "logical_training_tokens": 39000,
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / f"mixture_{seed}_{budget}.json"
    config.write_text(
        json.dumps(
            {
                "format_version": "vasu_pretraining_mixture_v1",
                "seed": seed,
                "sequence_length": 256,
                "token_budget": budget,
                "tokenizer_path": str(TOKENIZER),
                "tokenizer_sha256": _sha(TOKENIZER),
                "sources": [
                    {
                        "id": "fineweb_edu", "weight": 0.85,
                        "manifest": str(fine_manifest),
                        "manifest_sha256": _sha(fine_manifest), "logical_start": 0
                    },
                    {
                        "id": "wikimedia_factual", "weight": 0.15,
                        "path": str(wiki_train), "sha256": _sha(wiki_train),
                        "manifest": str(wiki_manifest),
                        "manifest_sha256": _sha(wiki_manifest)
                    },
                ],
                "output": {
                    "path": str(tmp_path / f"mixed_{seed}_{budget}.bin"),
                    "metadata": str(tmp_path / f"mixed_{seed}_{budget}.json"),
                    "schedule": str(tmp_path / f"mixed_{seed}_{budget}.schedule"),
                },
            }
        ),
        encoding="utf-8",
    )
    assert wiki_meta["train"]["token_count"] > 3 * 256
    return config


def test_approved_source_hash_validation(tmp_path: Path) -> None:
    _, source, _, _, _ = _prepare_small_wiki(tmp_path)
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        prepare_wikimedia_tokenized_split(
            source_path=source, expected_source_sha256="0" * 64,
            tokenizer_path=TOKENIZER, expected_tokenizer_sha256=_sha(TOKENIZER),
            output_train_path=tmp_path / "x.bin",
            output_validation_path=tmp_path / "y.bin",
            output_manifest_path=tmp_path / "z.json",
        )


def test_tokenizer_hash_and_vocabulary_are_validated(tmp_path: Path) -> None:
    _, source, _, _, _ = _prepare_small_wiki(tmp_path)
    with pytest.raises(ValueError, match="tokenizer SHA-256 mismatch"):
        prepare_wikimedia_tokenized_split(
            source_path=source, expected_source_sha256=_sha(source),
            tokenizer_path=TOKENIZER, expected_tokenizer_sha256="0" * 64,
            output_train_path=tmp_path / "x.bin",
            output_validation_path=tmp_path / "y.bin",
            output_manifest_path=tmp_path / "z.json",
        )


def test_parent_split_is_deterministic_and_leak_free() -> None:
    first = deterministic_parent_split(set(map(str, range(100))), validation_fraction=0.05, seed=42)
    second = deterministic_parent_split(set(map(str, range(100))), validation_fraction=0.05, seed=42)
    assert first == second
    assert not set(first[0]).intersection(first[1])
    assert len(first[1]) == 5


def test_tokenization_accounting_and_ids(tmp_path: Path) -> None:
    result, _, train, validation, _ = _prepare_small_wiki(tmp_path)
    actual = train.stat().st_size // 2 + validation.stat().st_size // 2
    assert actual == result["total_binary_tokens"]
    assert result["minimum_token_id"] >= 0
    assert result["maximum_token_id"] < 32000


def test_tokenization_is_byte_deterministic(tmp_path: Path) -> None:
    first, source, train, validation, _ = _prepare_small_wiki(tmp_path)
    hashes = (_sha(train), _sha(validation))
    second = prepare_wikimedia_tokenized_split(
        source_path=source, expected_source_sha256=_sha(source),
        tokenizer_path=TOKENIZER, expected_tokenizer_sha256=_sha(TOKENIZER),
        output_train_path=train, output_validation_path=validation,
        output_manifest_path=tmp_path / "second.json", seed=42,
    )
    assert hashes == (_sha(train), _sha(validation))
    assert first["train"]["token_count"] == second["train"]["token_count"]


def test_weights_allocate_exact_sequence_count() -> None:
    assert allocate_sequences(39062, {"fineweb_edu": 0.85, "wikimedia_factual": 0.15}) == {
        "fineweb_edu": 33203, "wikimedia_factual": 5859
    }


@pytest.mark.parametrize("weights", [{"a": 0.9, "b": 0.2}, {"a": 0.0, "b": 1.0}])
def test_invalid_weights_fail(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="weights"):
        allocate_sequences(10, weights)


def test_source_paths_and_hashes_are_validated(tmp_path: Path) -> None:
    config = _mixture_config(tmp_path)
    payload = json.loads(config.read_text())
    payload["sources"][1]["sha256"] = "0" * 64
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Wikimedia token hash mismatch"):
        build_mixture_artifact(config, repository_root=tmp_path)


def test_same_seed_same_schedule_and_different_seed_changes_it() -> None:
    counts = {"fineweb_edu": 85, "wikimedia_factual": 15}
    first = build_source_schedule(counts, seed=42)
    assert first == build_source_schedule(counts, seed=42)
    assert first != build_source_schedule(counts, seed=43)


def test_realized_mixture_and_exact_budget_alignment(tmp_path: Path) -> None:
    config = _mixture_config(tmp_path, budget=5200)
    metadata = build_mixture_artifact(config, repository_root=tmp_path)
    assert metadata["requested_token_budget"] == 5200
    assert metadata["actual_supervised_token_budget"] == 5120
    assert metadata["alignment_remainder_tokens"] == 80
    assert abs(metadata["sources"]["fineweb_edu"]["realized_percentage"] - 0.85) < 0.03


def test_resume_schedule_suffix_is_deterministic() -> None:
    schedule = build_source_schedule({"fineweb_edu": 85, "wikimedia_factual": 15}, seed=42)
    assert schedule[32:] == build_source_schedule(
        {"fineweb_edu": 85, "wikimedia_factual": 15}, seed=42
    )[32:]


def test_fineweb_and_wikimedia_records_are_sequence_compatible(tmp_path: Path) -> None:
    config = _mixture_config(tmp_path)
    metadata = build_mixture_artifact(config, repository_root=tmp_path)
    dataset = FixedRecordTokenDataset(
        metadata["output_path"], 256, expected_sha256=metadata["output_sha256"]
    )
    x, y = dataset[0]
    assert x.shape == y.shape == (256,)
    assert torch.equal(x[1:], y[:-1])


def test_fixed_record_resume_starts_at_requested_record(tmp_path: Path) -> None:
    path = tmp_path / "records.bin"
    np.arange(3 * 257, dtype=np.uint16).tofile(path)
    full = FixedRecordTokenDataset(path, 256)
    resumed = FixedRecordTokenDataset(path, 256, start_record=1)
    assert len(full) == 3 and len(resumed) == 2
    assert torch.equal(full[1][0], resumed[0][0])


def _training_config(tmp_path: Path, authorized: bool = False) -> Path:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"checkpoint")
    path = tmp_path / "training.json"
    payload = json.loads(
        Path("configs/training/vasu_60m_factual_cpt_wikimedia_15pct.json").read_text()
    )
    payload["training_authorized"] = authorized
    payload["starting_checkpoint"] = str(parent)
    payload["starting_checkpoint_sha256"] = _sha(parent)
    payload["output_directory"] = str(tmp_path / "output")
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parent_checkpoint_architecture_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class Shape:
        def __init__(self, *shape: int) -> None:
            self.shape = shape

    checkpoint = {
        "epoch": 0, "global_step": 200000, "optimizer": {}, "scheduler": {},
        "loss": 1.0,
        "model": {
            "embedding.embedding.weight": Shape(32000, 512),
            "lm_head.weight": Shape(32000, 512),
            "blocks.0.attention.q_proj.weight": Shape(512, 512),
            "blocks.9.mlp.w1.weight": Shape(2048, 512),
        },
    }
    monkeypatch.setattr(training, "sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(training.torch, "load", lambda *args, **kwargs: checkpoint)
    result = training._validate_parent_checkpoint(
        {"starting_checkpoint": "x.pt", "starting_checkpoint_sha256": "a" * 64,
         "parent_global_step": 200000}
    )
    assert result["global_step"] == 200000


def test_output_directory_cannot_overlap_parent(tmp_path: Path) -> None:
    path = _training_config(tmp_path)
    payload = json.loads(path.read_text())
    payload["output_directory"] = str(Path(payload["starting_checkpoint"]).parent)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="isolated"):
        training.load_experiment_config(path)


def test_training_authorization_false_blocks_before_updates(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="not authorized"):
        training.run_training(_training_config(tmp_path, authorized=False))
    assert not (tmp_path / "output").exists()


def test_legacy_text_dataset_remains_compatible(tmp_path: Path) -> None:
    path = tmp_path / "legacy.bin"
    np.arange(1024, dtype=np.uint16).tofile(path)
    dataset = TextDataset(path, seq_len=256)
    x, y = dataset[0]
    assert x.shape == y.shape == (256,)


def test_existing_generic_mixture_schema_remains_compatible() -> None:
    source = MixtureSource(
        id="general", domain="general", path="x.bin", format="token_bin",
        tokenizer_path="assets/tokenizer.json", token_count=100, weight=1.0,
        license="test", source_url="https://example.com", dataset_revision="v1",
        attribution_required=False, commercial_use_allowed=True, split="train",
        content_hash=None, deduplication_status="test", quality_filters=("test",), notes="",
    )
    manifest = MixtureManifest(
        manifest_version="1.0", experiment_id="legacy", description="legacy",
        tokenizer_path="assets/tokenizer.json", target_tokens=50, random_seed=42,
        sources=(source,), created_at="2026-01-01T00:00:00+00:00", notes="",
    )
    assert replace(manifest, target_tokens=40).sources == manifest.sources


def test_factual_evaluation_prompts_do_not_copy_release_sentences() -> None:
    prompts = json.loads(Path("evaluation/prompts_factual_cpt_v1.json").read_text())
    source = Path(
        "data/processed/pretrain/factual/wikimedia_pilot_release/documents.jsonl"
    ).read_text(encoding="utf-8").casefold()
    assert all(str(item["prompt"]).casefold() not in source for item in prompts)
