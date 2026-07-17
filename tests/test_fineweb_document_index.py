"""Focused tests for the document-level FineWeb deduplication index."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from scripts.build_fineweb_document_index import classify_artifacts
from vasu.data.deduplication.fineweb_index import (
    FineWebDocumentIndex,
    build_fineweb_document_index,
    load_index_config,
    sha256_file,
)
from vasu.data.deduplication.matching import MatchDecision, match_text
from vasu.data.deduplication.normalization import (
    NORMALIZATION_VERSION,
    jaccard_similarity,
    minhash_signature,
    normalize_for_matching,
    normalized_sha256,
    word_shingles,
)
from vasu.data.deduplication.schemas import (
    FineWebIndexConfig,
    FineWebSource,
    INDEX_FORMAT_VERSION,
    IndexRecord,
)


def write_jsonl(path: Path, texts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts),
        encoding="utf-8",
    )


def config_for(root: Path, texts: list[str], **changes: object) -> FineWebIndexConfig:
    source_path = root / "raw.jsonl"
    write_jsonl(source_path, texts)
    config = FineWebIndexConfig(
        format_version=INDEX_FORMAT_VERSION,
        output_path="index/fineweb.sqlite3",
        metadata_path="index/metadata.json",
        progress_path="index/progress.json",
        normalization_version=NORMALIZATION_VERSION,
        shingle_size=5,
        signature_size=64,
        bands=8,
        batch_size=1,
        maximum_documents=None,
        sources=(
            FineWebSource(
                source_id="fineweb",
                source_revision="revision",
                source_shard="shard-0",
                path="raw.jsonl",
                provenance_completeness="incomplete",
            ),
        ),
    )
    return replace(config, **changes)


def build(root: Path, texts: list[str]) -> tuple[FineWebIndexConfig, dict[str, object]]:
    config = config_for(root, texts)
    return config, build_fineweb_document_index(config, repository_root=root)


def test_artifact_inventory_classification(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "data/raw/pretrain/fineweb_1m.jsonl", ["text"])
    (tmp_path / "data/processed/pretrain").mkdir(parents=True)
    (tmp_path / "data/processed/pretrain/fineweb_1m.bin").write_bytes(b"\0\0")
    inventory = {item["path"]: item for item in classify_artifacts(tmp_path)}
    assert "document-level recoverable" in inventory["data/raw/pretrain/fineweb_1m.jsonl"]["classification"]
    assert inventory["data/processed/pretrain/fineweb_1m.bin"]["classification"] == "token-only and unsuitable"
    assert inventory["data/processed/pretrain/fineweb_extension_500m.bin"]["classification"] == "missing"


def test_index_schema_validation() -> None:
    record = IndexRecord(
        "doc", "source", "rev", "shard", None, "0" * 64, (1,) * 64,
        4, NORMALIZATION_VERSION, "incomplete", "raw.jsonl#line=1",
    )
    record.validate()
    with pytest.raises(ValueError, match="SHA-256"):
        replace(record, normalized_sha256="bad").validate()


def test_stable_normalization_preserves_unicode_and_punctuation() -> None:
    assert normalize_for_matching("  CAFÉ\r\nSecond\t line! ") == "café second line!"
    assert normalize_for_matching("café—test") == "café—test"


def test_normalization_version_mismatch(tmp_path: Path) -> None:
    config, _ = build(tmp_path, ["one two three four five six"])
    path = tmp_path / config.output_path
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value='other' WHERE key='normalization_version'")
    with pytest.raises(ValueError, match="normalization version mismatch"):
        FineWebDocumentIndex(path)


def test_exact_hash_matching(tmp_path: Path) -> None:
    text = "one two three four five six seven"
    config, _ = build(tmp_path, [text])
    index = FineWebDocumentIndex(tmp_path / config.output_path)
    result = match_text(index, text.upper(), chunk_id="wiki:1")
    index.close()
    assert result.match_type == "exact"
    assert result.decision == MatchDecision.REJECT
    assert result.fineweb_document_id == "fineweb:line:0"


def test_no_false_substring_match(tmp_path: Path) -> None:
    config, _ = build(tmp_path, ["alpha beta gamma delta epsilon zeta"])
    index = FineWebDocumentIndex(tmp_path / config.output_path)
    result = match_text(index, "alpha beta", chunk_id="wiki:1")
    index.close()
    assert result.match_type is None


def test_deterministic_shingle_signatures() -> None:
    text = "one two three four five six seven eight nine ten"
    assert minhash_signature(text) == minhash_signature(text)
    assert len(minhash_signature(text)) == 64


def test_near_duplicate_candidate_generation(tmp_path: Path) -> None:
    original = " ".join(f"token{index}" for index in range(100))
    changed = original.replace("token50", "replacement")
    config, _ = build(tmp_path, [original])
    index = FineWebDocumentIndex(tmp_path / config.output_path)
    result = match_text(index, changed, chunk_id="wiki:near", reject_threshold=0.75)
    index.close()
    assert result.match_type == "near"
    assert result.decision == MatchDecision.REJECT


def test_jaccard_threshold_behavior() -> None:
    left = word_shingles("one two three four five six seven eight")
    same = word_shingles("one two three four five six seven eight")
    other = word_shingles("red blue green black white orange purple yellow")
    assert jaccard_similarity(left, same) == 1.0
    assert jaccard_similarity(left, other) == 0.0


def test_duplicate_fineweb_records_are_counted_once(tmp_path: Path) -> None:
    _, metadata = build(tmp_path, ["same text repeated here five words", "same text repeated here five words"])
    assert metadata["indexed_documents"] == 1
    assert metadata["duplicate_hashes"] == 1


def test_incomplete_provenance_is_retained(tmp_path: Path) -> None:
    config, _ = build(tmp_path, ["one two three four five six"])
    index = FineWebDocumentIndex(tmp_path / config.output_path)
    result = index.exact(normalized_sha256("one two three four five six"))
    index.close()
    assert result and result["provenance_completeness"] == "incomplete"
    assert result["source_content_reference"] == "raw.jsonl#line=1"


def test_atomic_progress_writing(tmp_path: Path) -> None:
    config, _ = build(tmp_path, ["one two three four five six"])
    progress = tmp_path / config.progress_path
    assert progress.is_file()
    assert not progress.with_name(progress.name + ".tmp").exists()


def test_resume_without_duplicate_entries(tmp_path: Path) -> None:
    config = config_for(tmp_path, [f"document {index} has enough unique words here" for index in range(3)])
    with pytest.raises(InterruptedError):
        build_fineweb_document_index(
            config, repository_root=tmp_path, stop_after_documents=1
        )
    metadata = build_fineweb_document_index(config, repository_root=tmp_path, resume=True)
    assert metadata["indexed_documents"] == 3


def test_configuration_mismatch_rejected_on_resume(tmp_path: Path) -> None:
    config = config_for(tmp_path, ["one two three four five", "six seven eight nine ten"])
    with pytest.raises(InterruptedError):
        build_fineweb_document_index(config, repository_root=tmp_path, stop_after_documents=1)
    with pytest.raises(ValueError, match="configuration mismatch"):
        build_fineweb_document_index(
            replace(config, shingle_size=4), repository_root=tmp_path, resume=True
        )


def test_source_file_hash_mismatch(tmp_path: Path) -> None:
    config = config_for(tmp_path, ["one two three four five", "six seven eight nine ten"])
    with pytest.raises(InterruptedError):
        build_fineweb_document_index(config, repository_root=tmp_path, stop_after_documents=1)
    with (tmp_path / "raw.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": "changed source content"}) + "\n")
    with pytest.raises(ValueError, match="source-file hash mismatch"):
        build_fineweb_document_index(config, repository_root=tmp_path, resume=True)


def test_expected_source_hash_mismatch(tmp_path: Path) -> None:
    config = config_for(tmp_path, ["one two three four five"])
    source = replace(config.sources[0], expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="source-file hash mismatch"):
        build_fineweb_document_index(replace(config, sources=(source,)), repository_root=tmp_path)


def test_output_hash_validation_detects_tampering(tmp_path: Path) -> None:
    config, metadata = build(tmp_path, ["one two three four five six"])
    output = tmp_path / config.output_path
    assert sha256_file(output) == metadata["output_sha256"]
    with output.open("ab") as handle:
        handle.write(b"tamper")
    assert sha256_file(output) != metadata["output_sha256"]


def test_ambiguous_match_is_flagged(tmp_path: Path) -> None:
    original = " ".join(f"term{index}" for index in range(80))
    changed = original.replace("term20", "x").replace("term40", "y")
    config, _ = build(tmp_path, [original])
    index = FineWebDocumentIndex(tmp_path / config.output_path)
    result = match_text(
        index, changed, chunk_id="wiki:ambiguous", reject_threshold=1.0, review_threshold=0.5
    )
    index.close()
    assert result.decision == MatchDecision.REVIEW
    assert result.match_type == "ambiguous"


def test_bounded_index_is_not_training_eligible(tmp_path: Path) -> None:
    config = config_for(tmp_path, ["one two three four five", "six seven eight nine ten"], maximum_documents=1)
    metadata = build_fineweb_document_index(config, repository_root=tmp_path)
    assert metadata["completion_status"] == "bounded_complete"
    with pytest.raises(ValueError, match="bounded pilot"):
        FineWebDocumentIndex(tmp_path / config.output_path)


def test_unknown_or_invalid_config_values_rejected(tmp_path: Path) -> None:
    config = config_for(tmp_path, ["text"])
    with pytest.raises(ValueError, match="divisible"):
        replace(config, signature_size=63).validate()
    with pytest.raises(ValueError, match="duplicate"):
        replace(config, sources=(config.sources[0], config.sources[0])).validate()


def test_config_loader_and_no_cwd_assumption(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    loaded = load_index_config(root / "configs/data/deduplication/fineweb_index.json")
    assert loaded.normalization_version == NORMALIZATION_VERSION
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_fineweb_document_index.py"),
            "--config",
            str(root / "configs/data/deduplication/fineweb_index.json"),
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Full index build performed: no" in result.stdout


def test_invalid_jsonl_is_rejected(tmp_path: Path) -> None:
    config = config_for(tmp_path, ["valid document words one two three"])
    (tmp_path / "raw.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        build_fineweb_document_index(config, repository_root=tmp_path)


def test_empty_text_is_rejected_and_counted(tmp_path: Path) -> None:
    _, metadata = build(tmp_path, ["", "one two three four five"])
    assert metadata["indexed_documents"] == 1
    assert metadata["rejected_records"] == 1


def test_source_content_reference_is_stable(tmp_path: Path) -> None:
    config, _ = build(tmp_path, ["one two three four five six"])
    index = FineWebDocumentIndex(tmp_path / config.output_path)
    row = index.exact(hashlib.sha256(b"one two three four five six").hexdigest())
    index.close()
    assert row and row["source_content_reference"].endswith("#line=1")
