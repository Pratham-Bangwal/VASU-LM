from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from scripts.prepare_fineweb_extension import parse_args
from scripts.validate_fineweb_extension import validate_extension
from vasu.data.fineweb_extension import (
    ORIGINAL_TRAIN_END,
    ORIGINAL_VALIDATION_END,
    PREPROCESSING_VERSION,
    SEPARATOR,
    build_manifest,
    fingerprint_text,
    map_logical_training_offset,
    normalize_document,
    prepare_extension_from_rows,
    sha256_file,
    validate_manifest,
    validate_source_selection,
    validate_token_ids,
    write_uint16,
)
from vasu.tokenizer.tokenizer import VASUTokenizer


REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def tokenizer() -> VASUTokenizer:
    value = VASUTokenizer()
    value.load("assets/tokenizer.json")
    return value


def make_original_files(tmp_path: Path) -> tuple[Path, Path, str]:
    raw = tmp_path / "original.jsonl"
    raw.write_text(
        json.dumps({"text": "already present"}) + "\n",
        encoding="utf-8",
    )
    binary = tmp_path / "original.bin"
    np.asarray([1, 2, 3], dtype=np.uint16).tofile(binary)
    return raw, binary, sha256_file(binary)


def prepare(
    tmp_path: Path,
    rows: list[object],
    *,
    target: int = 20,
    resume: bool = False,
    stop_after_documents: int | None = None,
) -> tuple[Path, Path, dict | None]:
    raw, original, original_hash = make_original_files(tmp_path)
    output = tmp_path / "extension.bin"
    metadata = tmp_path / "extension_metadata.json"
    try:
        result = prepare_extension_from_rows(
            rows=rows,  # type: ignore[arg-type]
            tokenizer=tokenizer(),
            tokenizer_path=Path("assets/tokenizer.json"),
            output_path=output,
            metadata_path=metadata,
            dataset_repository="HuggingFaceFW/fineweb-edu",
            dataset_config="CC-MAIN-2025-26",
            dataset_revision=REVISION,
            split="train",
            target_tokens=target,
            buffer_token_limit=7,
            seed=42,
            resume=resume,
            max_documents=None,
            original_raw_path=raw,
            original_data_path=original,
            expected_original_sha256=original_hash,
            minimum_free_disk_gib=0,
            stop_after_documents=stop_after_documents,
        )
    except InterruptedError:
        result = None
    return output, metadata, result


def sample_rows() -> list[dict[str, str]]:
    return [
        {"id": "1", "text": "first educational document"},
        {"id": "2", "text": "second educational document"},
        {"id": "3", "text": "third educational document"},
        {"id": "4", "text": "fourth educational document"},
    ]


def test_tokenizer_hash_verification_and_metadata(tmp_path: Path):
    output, metadata_path, metadata = prepare(tmp_path, sample_rows())
    assert metadata is not None
    assert metadata["tokenizer_sha256"] == sha256_file(Path("assets/tokenizer.json"))
    assert metadata["preprocessing_version"] == PREPROCESSING_VERSION
    assert validate_extension(output, metadata_path)["actual_written_tokens"] >= 20


def test_uint16_writing_and_token_range(tmp_path: Path):
    output = tmp_path / "tokens.bin"
    with output.open("wb") as handle:
        assert write_uint16(handle, [0, 3, 31_999]) == 3
    assert np.fromfile(output, dtype=np.uint16).tolist() == [0, 3, 31_999]
    with pytest.raises(ValueError, match="outside"):
        validate_token_ids([32_000])


def test_buffer_flush_and_atomic_finalization(tmp_path: Path):
    output, _, _ = prepare(tmp_path, sample_rows(), target=28)
    assert output.exists()
    assert not output.with_suffix(".bin.tmp").exists()
    assert output.stat().st_size % 2 == 0


def test_resume_state_validation_rejects_identity_change(tmp_path: Path):
    output, _, result = prepare(
        tmp_path, sample_rows(), target=1_000, stop_after_documents=1
    )
    assert result is None
    raw = tmp_path / "original.jsonl"
    original = tmp_path / "original.bin"
    with pytest.raises(ValueError, match="dataset_config"):
        prepare_extension_from_rows(
            rows=sample_rows(),
            tokenizer=tokenizer(),
            tokenizer_path=Path("assets/tokenizer.json"),
            output_path=output,
            metadata_path=tmp_path / "metadata.json",
            dataset_repository="HuggingFaceFW/fineweb-edu",
            dataset_config="CC-MAIN-2024-10",
            dataset_revision=REVISION,
            split="train",
            target_tokens=1_000,
            buffer_token_limit=7,
            seed=42,
            resume=True,
            max_documents=None,
            original_raw_path=raw,
            original_data_path=original,
            expected_original_sha256=sha256_file(original),
            minimum_free_disk_gib=0,
        )


def test_interrupted_preparation_resumes_without_duplication(tmp_path: Path):
    rows = sample_rows()
    output, metadata_path, result = prepare(
        tmp_path, rows, target=28, stop_after_documents=1
    )
    assert result is None
    assert output.with_suffix(".bin.tmp").exists()
    raw = tmp_path / "original.jsonl"
    original = tmp_path / "original.bin"
    metadata = prepare_extension_from_rows(
        rows=rows,
        tokenizer=tokenizer(),
        tokenizer_path=Path("assets/tokenizer.json"),
        output_path=output,
        metadata_path=metadata_path,
        dataset_repository="HuggingFaceFW/fineweb-edu",
        dataset_config="CC-MAIN-2025-26",
        dataset_revision=REVISION,
        split="train",
        target_tokens=28,
        buffer_token_limit=7,
        seed=42,
        resume=True,
        max_documents=None,
        original_raw_path=raw,
        original_data_path=original,
        expected_original_sha256=sha256_file(original),
        minimum_free_disk_gib=0,
    )
    assert metadata["documents_written"] == len(rows)
    assert metadata["duplicates_within_extension"] == 0
    assert not output.with_suffix(".bin.state.json").exists()


def test_deduplication_persists_and_rejects_duplicates(tmp_path: Path):
    rows = [
        {"id": "1", "text": "duplicate"},
        {"id": "2", "text": " duplicate "},
        {"id": "3", "text": "unique enough for target"},
    ]
    _, _, metadata = prepare(tmp_path, rows, target=10)
    assert metadata is not None
    assert metadata["duplicates_within_extension"] == 1
    database = tmp_path / "extension.bin.dedup.sqlite3"
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT origin FROM fingerprints WHERE fingerprint = ?",
            (fingerprint_text(normalize_document("duplicate")),),
        ).fetchone()
    assert row == ("extension",)


def test_cross_shard_duplicate_empty_and_malformed_handling(tmp_path: Path):
    rows: list[object] = [
        {"id": "1", "text": "already present"},
        {"id": "2", "text": "  "},
        {"id": "3"},
        "not a mapping",
        {"id": "4", "text": "valid new content for output"},
    ]
    _, _, metadata = prepare(tmp_path, rows, target=5)
    assert metadata is not None
    assert metadata["cross_shard_exact_duplicates"] == 1
    assert metadata["empty_documents"] == 1
    assert metadata["malformed_documents"] == 2
    assert metadata["documents_dropped"] == 4


def test_file_size_and_metadata_are_consistent(tmp_path: Path):
    output, metadata_path, metadata = prepare(tmp_path, sample_rows())
    assert metadata is not None
    assert output.stat().st_size == metadata["actual_written_tokens"] * 2
    assert json.loads(metadata_path.read_text())["output_sha256"] == sha256_file(output)


def test_source_selection_requires_pinned_compatible_config():
    validate_source_selection(
        "HuggingFaceFW/fineweb-edu", "CC-MAIN-2025-26", REVISION, "train"
    )
    with pytest.raises(ValueError):
        validate_source_selection("HuggingFaceFW/fineweb", "default", "main", "train")
    with pytest.raises(ValueError, match="original source"):
        validate_source_selection(
            "HuggingFaceFW/fineweb-edu", "CC-MAIN-2013-20", REVISION, "train"
        )


def test_cli_supports_required_arguments():
    args = parse_args(
        [
            "--target-tokens",
            "1000000",
            "--output",
            "out.bin",
            "--resume",
            "--buffer-token-limit",
            "1234",
        ]
    )
    assert args.dataset_config == "CC-MAIN-2025-26"
    assert args.resume is True
    assert args.buffer_token_limit == 1234


def test_manifest_schema_boundaries_and_validation_preservation(tmp_path: Path):
    metadata = {
        "tokenizer_sha256": "abc",
        "actual_written_tokens": 500_000_000,
        "output_path": "data/processed/pretrain/fineweb_extension_500m.bin",
        "output_sha256": "def",
    }
    manifest = build_manifest(
        extension_metadata=metadata,
        manifest_path=tmp_path / "manifest.json",
    )
    validate_manifest(manifest)
    assert manifest["validation"]["start_token"] == ORIGINAL_TRAIN_END
    assert manifest["validation"]["end_token"] == ORIGINAL_VALIDATION_END


def test_logical_shard_boundary_mapping_and_capacity(tmp_path: Path):
    metadata = {
        "tokenizer_sha256": "abc",
        "actual_written_tokens": 500_000_000,
        "output_path": "extension.bin",
        "output_sha256": "def",
    }
    manifest = build_manifest(
        extension_metadata=metadata,
        manifest_path=tmp_path / "manifest.json",
    )
    assert map_logical_training_offset(manifest, 0)[1] == 0
    assert map_logical_training_offset(manifest, ORIGINAL_TRAIN_END - 1)[1] == ORIGINAL_TRAIN_END - 1
    assert map_logical_training_offset(manifest, ORIGINAL_TRAIN_END) == ("extension.bin", 0)
    assert map_logical_training_offset(manifest, ORIGINAL_TRAIN_END + 123) == ("extension.bin", 123)
    step_150000 = 150_000 * 2 * 16 * 256
    step_200000 = 200_000 * 2 * 16 * 256
    assert step_150000 == 1_228_800_000
    assert map_logical_training_offset(manifest, step_150000)[0].endswith("fineweb_1m.bin")
    assert map_logical_training_offset(manifest, step_200000) == (
        "extension.bin",
        step_200000 - ORIGINAL_TRAIN_END,
    )


def test_logical_offsets_never_map_into_validation(tmp_path: Path):
    metadata = {
        "tokenizer_sha256": "abc",
        "actual_written_tokens": 20,
        "output_path": "extension.bin",
        "output_sha256": "def",
    }
    manifest = build_manifest(
        extension_metadata=metadata,
        manifest_path=tmp_path / "manifest.json",
    )
    for offset in (ORIGINAL_TRAIN_END - 1, ORIGINAL_TRAIN_END, ORIGINAL_TRAIN_END + 19):
        path, physical = map_logical_training_offset(manifest, offset)
        assert not (
            path.endswith("fineweb_1m.bin")
            and ORIGINAL_TRAIN_END <= physical < ORIGINAL_VALIDATION_END
        )


def test_original_production_file_is_not_touched(tmp_path: Path):
    original = Path("data/processed/pretrain/fineweb_1m.bin")
    before = (original.stat().st_size, original.stat().st_mtime_ns)
    prepare(tmp_path, sample_rows())
    after = (original.stat().st_size, original.stat().st_mtime_ns)
    assert after == before


def test_complete_document_target_may_slightly_exceed(tmp_path: Path):
    _, _, metadata = prepare(
        tmp_path,
        [{"id": "1", "text": "one complete educational document"}],
        target=1,
    )
    assert metadata is not None
    assert metadata["actual_written_tokens"] > 1
    assert metadata["separator_token_ids"] == tokenizer().encode(SEPARATOR)


def test_small_smoke_preparation_is_deterministic(tmp_path: Path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_output, _, _ = prepare(first_dir, sample_rows(), target=20)
    second_output, _, _ = prepare(second_dir, sample_rows(), target=20)
    assert sha256_file(first_output) == sha256_file(second_output)
