"""Tests for the bounded Wikimedia factual preparation pipeline."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vasu.data.preparation.contamination import (
    check_contamination,
    load_prompt_evidence,
)
from vasu.data.preparation.chunking import (
    TextChunk,
    chunk_document,
    chunk_document_detailed,
    is_reference_section_heading,
)
from vasu.data.preparation.deduplication import (
    PilotDeduplicator,
    fineweb_document_index_status,
    load_fineweb_exact_hashes,
)
from vasu.data.preparation.filters import (
    clean_training_text,
    comparison_normalize,
    filter_wikimedia_record,
)
from vasu.data.preparation.quality import (
    _decode_candidate,
    assess_final_chunk,
    assess_text_quality,
    repair_mojibake,
)
from vasu.data.preparation.progress import load_progress, save_progress
from vasu.data.preparation.reporting import canonical_json_hash, sha256_file
from vasu.data.preparation.reporting import atomic_write_json
from vasu.data.preparation.schemas import (
    PreparationOutputPaths,
    PreparationProgress,
    WikimediaPreparationConfig,
)
from vasu.data.preparation.token_count import count_tokens
from vasu.data.preparation.wikimedia import (
    acquire_pinned_shard,
    config_to_dict,
    load_preparation_config,
    prepare_wikimedia_pilot,
    resolve_paths,
    iter_selected_parquet_rows,
    select_review_row_indices,
    summarize_review_diversity,
    validate_preparation_config,
    validate_preparation_output,
    validate_registry_approval,
)
from vasu.data.deduplication.fineweb_index import build_fineweb_document_index
from vasu.data.deduplication.schemas import (
    FineWebIndexConfig,
    FineWebSource,
    INDEX_FORMAT_VERSION,
)
from vasu.data.deduplication.normalization import NORMALIZATION_VERSION
from vasu.tokenizer.tokenizer import VASUTokenizer


PINNED_REVISION = "e6057dc557255a03c9c3c47ceab0eb44353b1bc5"
SHARD = "20231101.en/train-00000-of-00041.parquet"


class WordTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self._vocab:
                token_id = len(self._vocab) + 1
                self._vocab[word] = token_id
                self._inverse[token_id] = word
            ids.append(self._vocab[word])
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._inverse[token_id] for token_id in ids)


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def make_config(**changes: object) -> WikimediaPreparationConfig:
    paths = PreparationOutputPaths(
        raw_directory="data/raw/factual/wikimedia/20231101_en",
        interim_directory="data/interim/factual/wikimedia/20231101_en",
        processed_directory="data/processed/pretrain/factual/wikimedia_pilot",
        output_jsonl="data/processed/pretrain/factual/wikimedia_pilot/documents.jsonl",
        manifest_json="data/manifests/factual/wikimedia_pilot.json",
        progress_json="data/interim/factual/wikimedia/20231101_en/progress.json",
        summary_json="data/processed/pretrain/factual/wikimedia_pilot/summary.json",
        summary_text="data/processed/pretrain/factual/wikimedia_pilot/summary.txt",
    )
    config = WikimediaPreparationConfig(
        source_id="wikipedia_en_20231101_planned",
        dataset_name="wikimedia/wikipedia",
        subset="20231101.en",
        split="train",
        pinned_revision=PINNED_REVISION,
        shard_identifier=SHARD,
        tokenizer_path="assets/tokenizer.json",
        random_seed=42,
        max_download_bytes=500_000_000,
        max_raw_examples=100,
        max_accepted_parent_documents=20,
        max_accepted_chunks=20,
        max_output_tokens=20_000,
        minimum_document_characters=20,
        maximum_document_characters=10_000,
        exact_deduplication_enabled=True,
        near_deduplication_enabled=True,
        near_duplicate_similarity_threshold=0.9,
        contamination_check_enabled=True,
        contamination_ngram_words=5,
        chunking_enabled=True,
        target_chunk_tokens=768,
        maximum_chunk_tokens=1024,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=1,
        review_sampling_enabled=False,
        reference_section_behavior="keep",
        quality_warning_threshold=5,
        review_max_chunks_per_article=0,
        output_paths=paths,
        resume_enabled=True,
    )
    return replace(config, **changes)


def write_config(path: Path, config: WikimediaPreparationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config)), encoding="utf-8")


def article(index: int, text: str | None = None) -> dict[str, str]:
    return {
        "id": str(index),
        "url": f"https://en.wikipedia.org/wiki/Article_{index}",
        "title": f"Article {index}",
        "text": text or (
            f"Article {index} contains factual educational prose with enough "
            "distinct words for deterministic pilot processing and validation."
        ),
        "language": "en",
    }


def write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation/prompts.json").write_text(
        json.dumps([{"id": "gravity", "prompt": "Explain gravity in simple words"}]),
        encoding="utf-8",
    )
    # Pipeline behavior is tested independently of the already-tested registry gate.
    monkeypatch.setattr(
        "vasu.data.preparation.wikimedia.validate_registry_approval",
        lambda config, repository_root: None,
    )
    return tmp_path


def run_pipeline(
    repository: Path,
    rows: list[dict[str, object]],
    config: WikimediaPreparationConfig | None = None,
    **kwargs: object,
) -> tuple[dict[str, object], WikimediaPreparationConfig, Path]:
    selected = config or make_config()
    shard = repository / "input.parquet"
    write_parquet(shard, rows)
    manifest = prepare_wikimedia_pilot(
        selected,
        repository_root=repository,
        input_parquet=shard,
        tokenizer=WordTokenizer(),
        **kwargs,
    )
    return manifest, selected, shard


def install_fixed_chunker(
    monkeypatch: pytest.MonkeyPatch,
    chunks_per_parent: int,
) -> None:
    def fixed_chunks(text: str, *args: object, **kwargs: object) -> list[TextChunk]:
        parent_marker = "_".join(text.split()[:2])
        values = [
            f"{parent_marker} deterministic accepted chunk number {index}"
            for index in range(chunks_per_parent)
        ]
        return [TextChunk(value, len(value.split()), None) for value in values]

    monkeypatch.setattr(
        "vasu.data.preparation.wikimedia.chunk_document",
        fixed_chunks,
    )


def build_test_fineweb_index(repository: Path, texts: list[str]) -> str:
    source = repository / "fineweb.jsonl"
    source.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts),
        encoding="utf-8",
    )
    relative_index = "data/manifests/pretrain/fineweb_document_index.sqlite3"
    config = FineWebIndexConfig(
        format_version=INDEX_FORMAT_VERSION,
        output_path=relative_index,
        metadata_path="data/manifests/pretrain/fineweb_document_index_metadata.json",
        progress_path="data/interim/pretrain/fineweb_index_progress.json",
        normalization_version=NORMALIZATION_VERSION,
        shingle_size=5,
        signature_size=64,
        bands=8,
        batch_size=1,
        maximum_documents=None,
        sources=(
            FineWebSource(
                source_id="fineweb-test",
                source_revision="revision",
                source_shard="fixture",
                path="fineweb.jsonl",
            ),
        ),
    )
    build_fineweb_document_index(config, repository_root=repository)
    return relative_index


def test_valid_configuration_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_config(path, make_config())
    loaded = load_preparation_config(path)
    assert loaded == make_config()
    validate_preparation_config(loaded)


@pytest.mark.parametrize(
    "field",
    ["max_accepted_parent_documents", "max_accepted_chunks"],
)
def test_explicit_parent_and_chunk_limits_are_required(
    tmp_path: Path,
    field: str,
) -> None:
    payload = asdict(make_config())
    payload.pop(field)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"missing fields.*{field}"):
        load_preparation_config(path)


@pytest.mark.parametrize("value", [True, "20", 20.5])
def test_limit_types_reject_booleans_strings_and_fractions(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        validate_preparation_config(
            replace(make_config(), max_accepted_chunks=value)  # type: ignore[arg-type]
        )


def test_legacy_chunk_limit_maps_with_deprecation_warning(tmp_path: Path) -> None:
    payload = asdict(make_config())
    payload.pop("max_accepted_parent_documents")
    payload.pop("max_accepted_chunks")
    payload["max_accepted_documents"] = 17
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="interpreted as 'max_accepted_chunks'"):
        config = load_preparation_config(path)
    assert config.max_accepted_chunks == 17
    assert config.max_accepted_parent_documents == 17
    assert "max_accepted_documents" not in config_to_dict(config)


def test_legacy_and_explicit_chunk_limits_conflict(tmp_path: Path) -> None:
    payload = asdict(make_config())
    payload["max_accepted_documents"] = 20
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="either.*not both"):
        load_preparation_config(path)


def test_resolved_configuration_hash_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_config(path, make_config())
    first = load_preparation_config(path)
    second = load_preparation_config(path)
    assert canonical_json_hash(config_to_dict(first)) == canonical_json_hash(
        config_to_dict(second)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_download_bytes", 1_000_000_001),
        ("max_raw_examples", 10_001),
        ("max_accepted_parent_documents", 2_001),
        ("max_accepted_chunks", 4_001),
        ("max_output_tokens", 2_000_001),
        ("max_output_tokens", 0),
    ],
)
def test_invalid_limits_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="limit|positive|greater than zero"):
        validate_preparation_config(replace(make_config(), **{field: value}))


def test_pinned_revision_and_bounded_shard_are_enforced() -> None:
    with pytest.raises(ValueError, match="revision"):
        validate_preparation_config(make_config(pinned_revision="main"))
    with pytest.raises(ValueError, match="one explicit"):
        validate_preparation_config(make_config(shard_identifier="20231101.en/*.parquet"))


def test_unapproved_source_is_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "configs/data/sources", tmp_path / "configs/data/sources")
    (tmp_path / "configs/data").mkdir(exist_ok=True)
    shutil.copy2(
        root / "configs/data/vasu_60m_factual_pilot.json",
        tmp_path / "configs/data/vasu_60m_factual_pilot.json",
    )
    record_path = tmp_path / "configs/data/sources/wikimedia.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["approval_status"] = "pending"
    record["reviewed_by"] = ""
    record["reviewed_at"] = ""
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="approved|not training-ready"):
        validate_registry_approval(make_config(), tmp_path)


def test_dry_run_performs_no_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.prepare_wikimedia_pilot as cli

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        cli,
        "acquire_pinned_shard",
        lambda *args, **kwargs: pytest.fail("dry-run attempted acquisition"),
    )
    assert cli.main(["--config", str(root / "configs/data/preparation/wikimedia_pilot.json"), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Maximum accepted parent documents: 2,000" in output
    assert "Maximum accepted chunks: 4,000" in output


def test_acquisition_uses_one_pinned_shard(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = repository / "provider.parquet"
    write_parquet(source, [article(1)])
    calls: list[tuple[str, str]] = []

    class FakeApi:
        def dataset_info(self, repo_id: str, *, revision: str, files_metadata: bool):
            assert files_metadata
            calls.append((repo_id, revision))
            return SimpleNamespace(
                siblings=[SimpleNamespace(rfilename=SHARD, size=source.stat().st_size)]
            )

    def fake_download(**kwargs: object) -> str:
        assert kwargs["filename"] == SHARD
        assert kwargs["revision"] == PINNED_REVISION
        target = Path(str(kwargs["local_dir"])) / SHARD
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return str(target)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    result, metadata = acquire_pinned_shard(make_config(), repository)
    assert result.name == Path(SHARD).name
    assert calls == [("wikimedia/wikipedia", PINNED_REVISION)]
    assert metadata["input_sha256"] == sha256_file(source)


def test_acquisition_reuses_verified_cached_shard_without_network(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config()
    paths = resolve_paths(config, repository)
    cached = paths["raw_directory"] / SHARD
    write_parquet(cached, [article(1)])
    metadata = {
        "dataset_name": config.dataset_name,
        "subset": config.subset,
        "split": config.split,
        "pinned_revision": config.pinned_revision,
        "shard_identifier": config.shard_identifier,
        "downloaded_size_bytes": cached.stat().st_size,
        "input_sha256": sha256_file(cached),
    }
    paths["interim_directory"].mkdir(parents=True)
    (paths["interim_directory"] / "acquisition.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: pytest.fail("verified cache attempted network access"),
    )
    result, loaded = acquire_pinned_shard(config, repository)
    assert result == cached
    assert loaded == metadata


def test_required_parquet_schema_is_validated(repository: Path) -> None:
    path = repository / "bad.parquet"
    write_parquet(path, [{"id": "1", "title": "Missing fields"}])
    with pytest.raises(ValueError, match="required fields"):
        prepare_wikimedia_pilot(
            make_config(), repository_root=repository, input_parquet=path, tokenizer=WordTokenizer()
        )


@pytest.mark.parametrize(
    ("record", "reason"),
    [
        ({"id": "1", "title": "A", "text": "", "url": "x"}, "empty_text"),
        ({"id": "", "title": "A", "text": "long enough prose here", "url": "x"}, "missing_identity"),
        ({**article(1), "language": "fr"}, "non_english"),
        ({**article(1), "text": "#REDIRECT [[Elsewhere]]"}, "redirect"),
        ({**article(1), "title": "Mercury (disambiguation)"}, "disambiguation"),
        ({**article(1), "text": "tiny"}, "too_short"),
        ({**article(1), "text": "1234567890123456789012345"}, "no_alphabetic_text"),
    ],
)
def test_deterministic_filter_reason_codes(record: dict[str, object], reason: str) -> None:
    result = filter_wikimedia_record(record, minimum_characters=20, maximum_characters=1_000)
    assert not result.accepted
    assert result.reason == reason


def test_unicode_normalization_is_deterministic() -> None:
    assert comparison_normalize("  CAFÉ\r\nSecond   line ") == comparison_normalize(
        "cafe\u0301\nSecond line"
    )


def test_exact_and_near_duplicate_detection() -> None:
    dedup = PilotDeduplicator(exact_enabled=True, near_enabled=True, threshold=0.4)
    text = "one two three four five six seven eight nine"
    reason, fingerprint = dedup.classify(text)
    assert reason is None
    dedup.accept(text, fingerprint)
    assert dedup.classify(text)[0] == "exact_duplicate"
    assert dedup.classify("one two three four five six seven eight ten")[0] == "near_duplicate"


def test_prompt_contamination_detection(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps([{"id": "p1", "prompt": "Explain gravity in simple words"}]),
        encoding="utf-8",
    )
    evidence = load_prompt_evidence(prompts, 5)
    exact, matches = check_contamination(
        "Preface. Explain gravity in simple words. Suffix.", "doc", evidence
    )
    assert exact
    assert matches[0]["matching_method"] == "full_prompt"


def test_exact_tokenizer_count_uses_encode() -> None:
    assert count_tokens(WordTokenizer(), "one two three") == 3


def test_pipeline_limits_provenance_hashes_and_rejections(repository: Path) -> None:
    rows = [article(1), article(2, "tiny"), article(3)]
    config = make_config(max_output_tokens=18)
    manifest, _, shard = run_pipeline(repository, rows, config)
    paths = resolve_paths(config, repository)
    records = [json.loads(line) for line in paths["output_jsonl"].read_text(encoding="utf-8").splitlines()]
    assert manifest["completion_status"] == "token_limit"
    assert manifest["rejection_reasons"]["too_short"] == 1
    assert manifest["input_sha256"] == sha256_file(shard)
    assert manifest["output_artifact_hashes"]["documents_jsonl_sha256"] == sha256_file(paths["output_jsonl"])
    assert records[0]["provenance_metadata"]["source_row_index"] == 0
    assert records[0]["source_revision"] == PINNED_REVISION


def test_parent_and_chunk_limits_are_counted_independently(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fixed_chunker(monkeypatch, 5)
    config = make_config(
        max_accepted_parent_documents=2,
        max_accepted_chunks=7,
        max_output_tokens=20_000,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(repository, [article(1), article(2)], config)
    assert manifest["accepted_parent_documents"] == 2
    assert manifest["accepted_chunks"] == 7
    assert manifest["completion_status"] == "accepted_chunk_limit"


def test_parent_limit_stops_before_accepting_a_new_parent(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fixed_chunker(monkeypatch, 5)
    config = make_config(
        max_accepted_parent_documents=2,
        max_accepted_chunks=100,
        max_output_tokens=20_000,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(
        repository,
        [article(1), article(2), article(3)],
        config,
    )
    assert manifest["accepted_parent_documents"] == 2
    assert manifest["accepted_chunks"] == 10
    assert manifest["completion_status"] == "parent_document_limit"


def test_raw_example_limit_is_independent(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fixed_chunker(monkeypatch, 1)
    config = make_config(
        max_raw_examples=2,
        max_accepted_parent_documents=10,
        max_accepted_chunks=10,
        max_output_tokens=20_000,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(
        repository,
        [article(1), article(2), article(3)],
        config,
    )
    assert manifest["raw_examples"] == 2
    assert manifest["accepted_parent_documents"] == 2
    assert manifest["accepted_chunks"] == 2
    assert manifest["completion_status"] == "raw_example_limit"


def test_atomic_progress_write_and_load(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress = PreparationProgress("hash", "source", "revision", "shard")
    save_progress(path, progress)
    assert load_progress(path).configuration_hash == "hash"
    assert not list(tmp_path.glob("*.tmp"))


def test_legacy_progress_fails_without_unsafe_counter_migration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.json"
    path.write_text(
        json.dumps(
            {
                "configuration_hash": "hash",
                "source_id": "source",
                "pinned_revision": "revision",
                "shard_identifier": "shard",
                "accepted_documents": 4,
                "output_tokens": 100,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot be migrated safely"):
        load_progress(path)


def test_atomic_write_retries_transient_windows_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vasu.data.preparation.reporting as reporting

    real_replace = reporting.os.replace
    attempts = 0

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("synthetic transient lock")
        real_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", flaky_replace)
    path = tmp_path / "atomic.json"
    atomic_write_json(path, {"valid": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"valid": True}
    assert attempts == 3


def test_resume_has_no_duplicates_and_rejects_config_mismatch(repository: Path) -> None:
    rows = [article(1), article(2), article(3)]
    config = make_config()
    shard = repository / "input.parquet"
    write_parquet(shard, rows)
    with pytest.raises(InterruptedError):
        prepare_wikimedia_pilot(
            config,
            repository_root=repository,
            input_parquet=shard,
            tokenizer=WordTokenizer(),
            stop_after_rows=1,
        )
    with pytest.raises(ValueError, match="configuration"):
        prepare_wikimedia_pilot(
            replace(config, max_raw_examples=99),
            repository_root=repository,
            input_parquet=shard,
            tokenizer=WordTokenizer(),
            resume=True,
        )
    manifest = prepare_wikimedia_pilot(
        config,
        repository_root=repository,
        input_parquet=shard,
        tokenizer=WordTokenizer(),
        resume=True,
    )
    records = resolve_paths(config, repository)["output_jsonl"].read_text(encoding="utf-8").splitlines()
    assert len(records) == manifest["accepted_chunks"] == 3
    assert manifest["accepted_parent_documents"] == 3
    assert len({json.loads(line)["document_id"] for line in records}) == 3
    progress = load_progress(resolve_paths(config, repository)["progress_json"])
    assert progress.accepted_chunks == 3
    assert progress.accepted_parent_documents == 3
    assert len(set(progress.accepted_parent_document_ids or [])) == 3


def test_restart_is_clean_and_deterministic(repository: Path) -> None:
    rows = [article(1), article(2), article(3)]
    first, config, shard = run_pipeline(repository, rows)
    output = resolve_paths(config, repository)["output_jsonl"]
    first_hash = sha256_file(output)
    second = prepare_wikimedia_pilot(
        config,
        repository_root=repository,
        input_parquet=shard,
        tokenizer=WordTokenizer(),
        restart=True,
    )
    assert sha256_file(output) == first_hash
    assert first["total_vasu_tokens"] == second["total_vasu_tokens"]
    assert first["accepted_parent_documents"] == second["accepted_parent_documents"]
    assert first["accepted_chunks"] == second["accepted_chunks"]


def test_fineweb_cross_dedup_blocked_without_document_index(tmp_path: Path) -> None:
    status = fineweb_document_index_status(tmp_path)
    assert status["status"] == "blocked"
    assert "document-level index" in str(status["required_artifact"])


def test_fineweb_document_hash_index_is_loaded(tmp_path: Path) -> None:
    index = tmp_path / "data/processed/pretrain/fineweb_document_hashes.txt"
    index.parent.mkdir(parents=True)
    expected = hashlib.sha256(b"document").hexdigest()
    index.write_text(expected + "\n", encoding="utf-8")
    hashes, status = load_fineweb_exact_hashes(tmp_path)
    assert hashes == {expected}
    assert status["status"] == "available"
    assert status["loaded_exact_hashes"] == 1


def test_wikimedia_exact_fineweb_match_is_rejected(repository: Path) -> None:
    text = "This exact factual article has enough distinct words for matching correctly."
    index_path = build_test_fineweb_index(repository, [text])
    config = make_config(
        chunking_enabled=False,
        fineweb_index_path=index_path,
        fineweb_index_required=True,
    )
    manifest, _, _ = run_pipeline(repository, [article(1, text)], config)
    assert manifest["accepted_chunks"] == 0
    assert manifest["accepted_parent_documents"] == 0
    assert manifest["rejection_reasons"]["fineweb_exact_duplicate"] == 1
    assert manifest["fineweb_overlap_matches"][0]["decision"] == "reject"


def test_wikimedia_near_fineweb_match_is_rejected(repository: Path) -> None:
    original = " ".join(f"factword{index}" for index in range(100))
    changed = original.replace("factword50", "replacementword")
    index_path = build_test_fineweb_index(repository, [original])
    config = make_config(
        chunking_enabled=False,
        fineweb_index_path=index_path,
        fineweb_index_required=True,
    )
    manifest, _, _ = run_pipeline(repository, [article(1, changed)], config)
    assert manifest["accepted_chunks"] == 0
    assert manifest["accepted_parent_documents"] == 0
    assert manifest["rejection_reasons"]["fineweb_near_duplicate"] == 1


def test_wikimedia_ambiguous_fineweb_match_is_flagged(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = " ".join(f"factword{index}" for index in range(80))
    changed = original.replace("factword40", "replacementword")
    index_path = build_test_fineweb_index(repository, [original])
    from vasu.data.deduplication import matching

    real_match = matching.match_text
    monkeypatch.setattr(
        "vasu.data.preparation.wikimedia.match_text",
        lambda index, text, chunk_id: real_match(
            index,
            text,
            chunk_id=chunk_id,
            reject_threshold=1.0,
            review_threshold=0.5,
        ),
    )
    config = make_config(
        chunking_enabled=False,
        fineweb_index_path=index_path,
        fineweb_index_required=True,
    )
    manifest, selected, _ = run_pipeline(repository, [article(1, changed)], config)
    assert manifest["accepted_chunks"] == 1
    output = json.loads(resolve_paths(selected, repository)["output_jsonl"].read_text())
    assert "fineweb_ambiguous_overlap" in output["quality_warnings"]
    assert output["fineweb_overlap"]["decision"] == "review"


def test_review_mode_without_index_remains_allowed(repository: Path) -> None:
    config = make_config().for_review_sample()
    manifest, _, _ = run_pipeline(repository, [article(1)], config)
    assert manifest["fineweb_cross_deduplication"]["status"] == "blocked"
    assert manifest["accepted_chunks"] == 1


def test_default_factual_mode_without_index_is_blocked(repository: Path) -> None:
    config = make_config(fineweb_index_required=True)
    with pytest.raises(RuntimeError, match="blocked"):
        run_pipeline(repository, [article(1)], config)


def test_output_validation_detects_tampering(repository: Path) -> None:
    _, config, _ = run_pipeline(repository, [article(1), article(2)])
    assert validate_preparation_output(
        config, repository_root=repository, tokenizer=WordTokenizer()
    )["valid"]
    output = resolve_paths(config, repository)["output_jsonl"]
    output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_preparation_output(
            config, repository_root=repository, tokenizer=WordTokenizer()
        )


def test_cli_has_no_current_working_directory_dependency(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/prepare_wikimedia_pilot.py"),
            "--config",
            str(root / "configs/data/preparation/wikimedia_pilot.json"),
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Download performed: no" in result.stdout


def test_generated_data_paths_are_gitignored() -> None:
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    assert "data/interim/" in ignored
    assert "data/manifests/factual/" in ignored
    assert "data/raw/" in ignored
    assert "data/processed/" in ignored


def mojibake(value: str, encoding: str = "cp1252") -> str:
    return value.encode("utf-8").decode(encoding)


def test_clean_unicode_remains_unchanged() -> None:
    clean = "Enragés – μm m−2 ‘quoted’ 23° Târgoviște Ãngela Île"
    result = repair_mojibake(clean)
    assert result.text == clean
    assert not result.repaired
    assert result.repair_count == 0


@pytest.mark.parametrize(
    "clean",
    ["Enragés", "1756–1836", "μm", "m−2", "it’s", "23°"],
)
def test_common_mojibake_is_repaired(clean: str) -> None:
    encoding = "latin-1" if clean == "m−2" else "cp1252"
    result = repair_mojibake(mojibake(clean, encoding))
    assert result.text == clean
    assert result.repaired
    assert result.rejection_reason is None


def test_double_mojibake_is_repaired() -> None:
    first = mojibake("Enragés")
    second = mojibake(first)
    result = repair_mojibake(second)
    assert result.text == "Enragés"
    assert result.repair_count == 2


def test_low_confidence_corruption_is_flagged() -> None:
    result = repair_mojibake("Suspicious Ã© marker with valid Ω")
    assert result.text == "Suspicious Ã© marker with valid Ω"
    assert result.rejection_reason == "low_confidence_encoding_corruption"


def test_replacement_and_control_characters_are_rejected() -> None:
    assert repair_mojibake("bad \ufffd text").rejection_reason == "replacement_character"
    assert assess_text_quality("bad\x01text")[1] == "control_character"


@pytest.mark.parametrize(
    ("raw", "expected", "metadata_key"),
    [
        ("word[1] next", "word next", "citations_removed"),
        ("end<ref>source</ref>of", "end of", "references_removed"),
        ("text{{template}}continuation", "text continuation", "templates_removed"),
        ("word[1], next", "word, next", "citations_removed"),
    ],
)
def test_inline_cleanup_preserves_word_boundaries(
    raw: str, expected: str, metadata_key: str
) -> None:
    cleaned, metadata, rejection = clean_training_text(raw)
    assert rejection is None
    assert cleaned == expected
    assert metadata[metadata_key] == 1


def test_joined_word_smoke_regressions_are_fixed() -> None:
    raw = (
        "for<ref>x</ref>authority end{{x}}of within[1]anarchist "
        "from<ref/>the as{{t}}distinct"
    )
    cleaned, _, rejection = clean_training_text(raw)
    assert rejection is None
    assert cleaned == "for authority end of within anarchist from the as distinct"


@pytest.mark.parametrize(
    ("raw", "expected_value"),
    [
        ("Ammonia has the formula {{chem2|NH3}}.", "NH3"),
        ("The rectangle uses {{math|area = base × height}}.", "area = base × height"),
        ("The aircraft reached {{convert|11850|km/h|mph}}.", "11850 km/h"),
        (
            "The commission is {{lang|ca|Comissió de Toponímia}}.",
            "Comissió de Toponímia",
        ),
        ("The text states {{quote|First principle}}.", "First principle"),
        (
            "Einstein's ''Zur Elektrodynamik bewegter Körper'' was published.",
            "Zur Elektrodynamik bewegter Körper",
        ),
    ],
)
def test_readable_wikimedia_markup_values_are_preserved(
    raw: str,
    expected_value: str,
) -> None:
    cleaned, metadata, rejection = clean_training_text(raw)
    assert rejection is None
    assert expected_value in cleaned
    if "{{" in raw:
        assert metadata["templates_preserved"] == 1
        assert metadata["templates_removed"] == 0


def test_structured_enumeration_template_is_preserved_as_lines() -> None:
    cleaned, metadata, rejection = clean_training_text(
        "The following principles:\n{{ubl|First principle|Second principle}}"
    )
    assert rejection is None
    assert cleaned == "The following principles:\nFirst principle\nSecond principle"
    assert metadata["templates_preserved"] == 1


def test_valid_greek_math_and_accents_survive_cleanup() -> None:
    raw = "Enragés measured 0.5 μm at 23° and reported m−2."
    cleaned, metadata, rejection = clean_training_text(raw)
    assert cleaned == raw
    assert rejection is None
    assert not metadata["encoding_repaired"]


@pytest.mark.parametrize(
    "clean",
    [
        "al-ʿarabiyyah العربية qāf š ḍād",
        "14 March 1879\u00a0– 18 April 1955 — biography",
        "Δv ≈ 5.2 km·s⁻¹ and μm²",
        "René Descartes, Bahá'í, Enragés, São Paulo",
    ],
)
def test_failed_single_byte_repair_candidates_leave_unicode_untouched(
    clean: str,
) -> None:
    assert _decode_candidate(clean, "cp1252") is None
    assert _decode_candidate(clean, "latin-1") is None
    result = repair_mojibake(clean)
    assert result.text == clean
    assert not result.repaired
    assert "\ufffd" not in result.text


@pytest.mark.parametrize(
    "source_style_text",
    [
        "Abacus terminology uses suanpan, ṣaḥīfa, and ʾabāq — without loss.",
        "Albert Einstein lived from 14 March 1879\u00a0– 18 April 1955; E = mc².",
        "Asteroid motion may use Δv ≈ 5.2 km·s⁻¹ — an orbital quantity.",
        "Arabic transliteration includes al-ʿarabiyyah, qāf, š, and ḍād.",
    ],
)
def test_source_style_unicode_survives_real_token_chunking(
    source_style_text: str,
) -> None:
    tokenizer = VASUTokenizer()
    tokenizer.load(str(Path(__file__).resolve().parents[1] / "assets/tokenizer.json"))
    text = " ".join([source_style_text] * 40)
    chunks = chunk_document(
        text,
        tokenizer,
        target_tokens=48,
        maximum_tokens=64,
        minimum_tokens=8,
        overlap_tokens=0,
    )
    reconstructed = "".join("".join(chunk.text.split()) for chunk in chunks)
    assert reconstructed == "".join(text.split())
    assert all("\ufffd" not in chunk.text for chunk in chunks)
    assert all(chunk.token_count <= 64 for chunk in chunks)


def test_chunking_is_deterministic_and_bounded() -> None:
    tokenizer = WordTokenizer()
    text = "\n\n".join(
        " ".join(f"p{paragraph}_{word}" for word in range(7))
        for paragraph in range(8)
    )
    kwargs = dict(target_tokens=16, maximum_tokens=20, minimum_tokens=5, overlap_tokens=2)
    first = chunk_document(text, tokenizer, **kwargs)
    second = chunk_document(text, tokenizer, **kwargs)
    assert first == second
    assert len(first) >= 3
    assert max(chunk.token_count for chunk in first) <= 20


def test_chunking_prefers_paragraph_boundaries() -> None:
    tokenizer = WordTokenizer()
    first = " ".join(f"first{i}" for i in range(8))
    second = " ".join(f"second{i}" for i in range(8))
    chunks = chunk_document(
        f"{first}\n\n{second}",
        tokenizer,
        target_tokens=8,
        maximum_tokens=10,
        minimum_tokens=3,
        overlap_tokens=0,
    )
    assert [chunk.text for chunk in chunks] == [first, second]


def test_small_trailing_chunk_merges_when_safe() -> None:
    tokenizer = WordTokenizer()
    chunks = chunk_document(
        "one two three four five six\n\nseven eight",
        tokenizer,
        target_tokens=6,
        maximum_tokens=8,
        minimum_tokens=3,
        overlap_tokens=0,
    )
    assert len(chunks) == 1
    assert chunks[0].token_count == 8


def test_chunk_output_provenance_hashes_and_limits(repository: Path) -> None:
    text = "\n\n".join(
        " ".join(f"section{section}_word{word}" for word in range(8))
        for section in range(5)
    )
    config = make_config(
        target_chunk_tokens=10,
        maximum_chunk_tokens=12,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=1,
    )
    manifest, _, _ = run_pipeline(repository, [article(7, text)], config)
    records = [
        json.loads(line)
        for line in resolve_paths(config, repository)["output_jsonl"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["format_version"] == "wikimedia_pilot_v4"
    assert manifest["accepted_chunks"] == len(records) >= 3
    assert manifest["accepted_parent_documents"] == 1
    assert "accepted_documents" not in manifest
    summary_text = resolve_paths(config, repository)["summary_text"].read_text(
        encoding="utf-8"
    )
    assert "Accepted parent documents: 1" in summary_text
    assert f"Accepted chunks: {len(records)}" in summary_text
    for index, record in enumerate(records):
        assert record["format_version"] == "wikimedia_pilot_document_v3"
        assert record["boundary_start_type"] in {
            "section",
            "paragraph",
            "sentence",
            "word_fallback",
            "token_fallback",
        }
        assert record["boundary_end_type"] in {
            "section",
            "paragraph",
            "sentence",
            "word_fallback",
            "token_fallback",
        }
        assert record["overlap_characters"] >= 0
        assert record["parent_document_id"] == "7"
        assert record["chunk_id"] == f"7:{index:04d}"
        assert record["chunk_index"] == index
        assert record["chunk_count"] == len(records)
        assert record["provenance_metadata"]["source_row_index"] == 0
        assert record["token_count"] <= 12
        assert record["normalized_sha256"] == hashlib.sha256(
            comparison_normalize(record["cleaned_text"]).encode("utf-8")
        ).hexdigest()


def test_deduplication_operates_on_final_chunks(repository: Path) -> None:
    text = " ".join(f"same{index}" for index in range(30))
    config = make_config(
        target_chunk_tokens=10,
        maximum_chunk_tokens=10,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=0,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(repository, [article(1, text), article(2, text)], config)
    assert manifest["accepted_chunks"] == 3
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["exact_duplicates"] == 3


def test_contamination_operates_on_final_chunks(repository: Path) -> None:
    text = (
        "Explain gravity in simple words and include enough surrounding factual prose.\n\n"
        "A separate clean paragraph contains educational content for this pilot record."
    )
    config = make_config(
        target_chunk_tokens=10,
        maximum_chunk_tokens=14,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=0,
    )
    manifest, _, _ = run_pipeline(repository, [article(1, text)], config)
    assert manifest["rejection_reasons"]["contamination_exact_prompt"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["accepted_parent_documents"] == 1


def test_token_limit_accounting_uses_chunks(repository: Path) -> None:
    text = "\n\n".join(
        " ".join(f"part{part}_{word}" for word in range(6)) for part in range(4)
    )
    config = make_config(
        max_output_tokens=10,
        target_chunk_tokens=6,
        maximum_chunk_tokens=6,
        minimum_chunk_tokens=2,
        chunk_overlap_tokens=0,
    )
    manifest, _, _ = run_pipeline(repository, [article(1, text)], config)
    assert manifest["completion_status"] == "token_limit"
    assert manifest["accepted_chunks"] == 1
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["total_vasu_tokens"] == 6


def test_final_chunk_quality_gate_rejects_replacement_and_keeps_clean_sibling(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clean = "A clean sibling chunk remains eligible for factual preparation."
    corrupt = "An introduced replacement \ufffd must be rejected."
    tokenizer = WordTokenizer()
    monkeypatch.setattr(
        "vasu.data.preparation.wikimedia.chunk_document",
        lambda *args, **kwargs: [
            TextChunk(corrupt, len(tokenizer.encode(corrupt)), None),
            TextChunk(clean, len(tokenizer.encode(clean)), None),
        ],
    )
    config = make_config(
        chunking_enabled=True,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(
        repository,
        [article(1)],
        config,
    )
    records = [
        json.loads(line)
        for line in resolve_paths(config, repository)["output_jsonl"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert manifest["rejection_reasons"]["replacement_character"] == 1
    assert manifest["quality_rejections"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["total_vasu_tokens"] == len(tokenizer.encode(clean))
    assert [record["cleaned_text"] for record in records] == [clean]


def test_source_replacement_character_is_rejected_before_chunking(
    repository: Path,
) -> None:
    clean = "A separate clean source article remains eligible for preparation."
    config = make_config(
        chunking_enabled=False,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(
        repository,
        [article(1, "Source text containing \ufffd is corrupt."), article(2, clean)],
        config,
    )
    assert manifest["rejection_reasons"]["replacement_character"] == 1
    assert manifest["quality_rejections"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["accepted_parent_documents"] == 1


def test_chunked_resume_is_deterministic(repository: Path) -> None:
    text = "\n\n".join(
        " ".join(f"part{part}_{word}" for word in range(8)) for part in range(5)
    )
    config = make_config(
        target_chunk_tokens=8,
        maximum_chunk_tokens=9,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=1,
    )
    shard = repository / "input.parquet"
    write_parquet(shard, [article(1, text)])
    with pytest.raises(InterruptedError):
        prepare_wikimedia_pilot(
            config,
            repository_root=repository,
            input_parquet=shard,
            tokenizer=WordTokenizer(),
            stop_after_rows=1,
        )
    manifest = prepare_wikimedia_pilot(
        config,
        repository_root=repository,
        input_parquet=shard,
        tokenizer=WordTokenizer(),
        resume=True,
    )
    records = resolve_paths(config, repository)["output_jsonl"].read_text(encoding="utf-8").splitlines()
    assert len(records) == manifest["accepted_chunks"]
    assert manifest["accepted_parent_documents"] == 1
    assert len({json.loads(line)["chunk_id"] for line in records}) == len(records)


def test_default_mode_can_exceed_two_thousand_chunks(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fixed_chunker(monkeypatch, 2_001)
    monkeypatch.setattr(
        "vasu.data.preparation.wikimedia.save_progress",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("vasu.data.preparation.wikimedia.os.fsync", lambda *args: None)
    config = make_config(
        max_accepted_parent_documents=20,
        max_accepted_chunks=4_000,
        max_output_tokens=20_000,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(repository, [article(1)], config)
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["accepted_chunks"] == 2_001
    assert manifest["completion_status"] == "source_exhausted"


def test_previous_v1_output_is_rejected_clearly(repository: Path) -> None:
    _, config, _ = run_pipeline(repository, [article(1)])
    manifest_path = resolve_paths(config, repository)["manifest_json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "wikimedia_pilot_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="expected wikimedia_pilot_v2 through wikimedia_pilot_v4"):
        validate_preparation_output(
            config, repository_root=repository, tokenizer=WordTokenizer()
        )


def test_review_row_selection_is_deterministic_unique_and_spread() -> None:
    random.seed(991)
    before = random.getstate()
    first = select_review_row_indices(156_289, 500, 42)
    second = select_review_row_indices(156_289, 500, 42)
    assert first == second
    assert len(first) == len(set(first)) == 500
    assert first[0] < 1_000
    assert first[-1] > 155_000
    assert random.getstate() == before


def test_selected_reader_does_not_use_full_table_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows.parquet"
    rows = [article(index) for index in range(30)]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=10)
    monkeypatch.setattr(
        pq,
        "read_table",
        lambda *args, **kwargs: pytest.fail("full table load attempted"),
    )
    selected = list(iter_selected_parquet_rows(path, (1, 15, 29)))
    assert [index for index, _ in selected] == [1, 15, 29]


def test_review_mode_paths_and_limits_do_not_overlap_smoke() -> None:
    base = make_config()
    smoke = base.for_smoke_test()
    review = base.for_review_sample()
    assert review.max_raw_examples == 100  # Base test fixture is already capped.
    assert review.max_accepted_parent_documents == 20
    assert review.max_accepted_chunks == 20
    assert review.max_output_tokens == 20_000
    assert review.review_sampling_enabled
    assert review.reference_section_behavior == "flag"
    assert review.review_max_chunks_per_article == 5
    assert review.output_paths.output_jsonl.endswith("documents_review.jsonl")
    assert review.output_paths != smoke.output_paths
    assert review.output_paths != base.output_paths


def test_production_review_mode_hard_defaults() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_preparation_config(
        root / "configs/data/preparation/wikimedia_pilot.json"
    ).for_review_sample()
    assert config.max_raw_examples == 500
    assert config.max_accepted_parent_documents == 50
    assert config.max_accepted_chunks == 50
    assert config.max_output_tokens == 50_000
    assert config.review_max_chunks_per_article == 5


def test_production_modes_apply_reference_section_policy() -> None:
    root = Path(__file__).resolve().parents[1]
    base = load_preparation_config(root / "configs/data/preparation/wikimedia_pilot.json")
    assert base.reference_section_behavior == "exclude"
    assert base.for_smoke_test().reference_section_behavior == "exclude"
    assert base.for_review_sample().reference_section_behavior == "flag"


def test_review_manifest_records_selected_and_inspected_rows(repository: Path) -> None:
    rows = [article(index) for index in range(30)]
    config = replace(
        make_config(),
        review_sampling_enabled=True,
        review_max_chunks_per_article=5,
        max_raw_examples=10,
        max_accepted_parent_documents=20,
        max_accepted_chunks=20,
    )
    manifest, _, _ = run_pipeline(repository, rows, config)
    selection = manifest["row_selection"]
    assert selection["mode"] == "deterministic_broad_review"
    assert len(selection["selected_row_indices"]) == 10
    assert selection["inspected_row_indices"] == selection["selected_row_indices"]


def test_review_resume_is_deterministic(repository: Path) -> None:
    rows = [article(index) for index in range(30)]
    config = replace(
        make_config(),
        review_sampling_enabled=True,
        review_max_chunks_per_article=5,
        max_raw_examples=10,
        max_accepted_parent_documents=20,
        max_accepted_chunks=20,
    )
    shard = repository / "input.parquet"
    write_parquet(shard, rows)
    with pytest.raises(InterruptedError):
        prepare_wikimedia_pilot(
            config,
            repository_root=repository,
            input_parquet=shard,
            tokenizer=WordTokenizer(),
            stop_after_rows=1,
        )
    manifest = prepare_wikimedia_pilot(
        config,
        repository_root=repository,
        input_parquet=shard,
        tokenizer=WordTokenizer(),
        resume=True,
    )
    selected = manifest["row_selection"]["selected_row_indices"]
    inspected = manifest["row_selection"]["inspected_row_indices"]
    assert inspected == selected
    assert len(inspected) == len(set(inspected))


def test_review_diversity_metrics_and_largest_article_warning(tmp_path: Path) -> None:
    output = tmp_path / "review.jsonl"
    records = []
    for index in range(4):
        parent = "dominant" if index < 3 else "other"
        records.append(
            {
                "parent_document_id": parent,
                "token_count": index + 2,
                "title": parent,
                "quality_warnings": [],
                "cleaned_text": "clean text",
                "provenance_metadata": {"source_row_index": index},
            }
        )
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = summarize_review_diversity(
        output, maximum_chunk_tokens=10, quality_warning_threshold=5
    )
    assert summary["distinct_parent_articles"] == 2
    assert summary["chunk_tokens"] == {
        "minimum": 2,
        "median": 3.5,
        "mean": 3.5,
        "maximum": 5,
    }
    assert summary["largest_article_proportion"] == 0.75
    assert "largest_article_exceeds_25_percent" in summary["warnings"]


def reference_article() -> dict[str, str]:
    return article(
        1,
        "A factual lead paragraph contains enough words for deterministic review.\n\n"
        "References\n\n"
        "A reference entry provides publication information and source details.",
    )


def test_reference_sections_are_flagged(repository: Path) -> None:
    config = replace(
        make_config(),
        reference_section_behavior="flag",
        target_chunk_tokens=15,
        maximum_chunk_tokens=20,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=0,
    )
    manifest, _, _ = run_pipeline(repository, [reference_article()], config)
    records = [
        json.loads(line)
        for line in resolve_paths(config, repository)["output_jsonl"]
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    flagged = [record for record in records if "reference_section" in record["quality_warnings"]]
    assert flagged
    assert all(record["section_title"] == "References" for record in flagged)
    assert manifest["reference_section_detections"] >= 1


def test_reference_sections_can_be_excluded(repository: Path) -> None:
    config = replace(
        make_config(),
        reference_section_behavior="exclude",
        target_chunk_tokens=15,
        maximum_chunk_tokens=20,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=0,
    )
    manifest, _, _ = run_pipeline(repository, [reference_article()], config)
    assert manifest["rejection_reasons"]["reference_section_excluded"] >= 1
    records = resolve_paths(config, repository)["output_jsonl"].read_text(encoding="utf-8")
    assert "reference entry" not in records.casefold()


def test_review_output_validation(repository: Path) -> None:
    config = replace(
        make_config(),
        review_sampling_enabled=True,
        review_max_chunks_per_article=5,
        max_raw_examples=5,
    )
    run_pipeline(repository, [article(index) for index in range(10)], config)
    result = validate_preparation_output(
        config, repository_root=repository, tokenizer=WordTokenizer()
    )
    assert result["valid"]


def test_review_mode_caps_chunks_per_article_for_diversity(repository: Path) -> None:
    text = " ".join(f"word{index}" for index in range(30))
    config = replace(
        make_config(),
        review_sampling_enabled=True,
        review_max_chunks_per_article=2,
        max_raw_examples=3,
        max_accepted_parent_documents=6,
        max_accepted_chunks=6,
        target_chunk_tokens=10,
        maximum_chunk_tokens=10,
        minimum_chunk_tokens=3,
        chunk_overlap_tokens=0,
    )
    manifest, _, _ = run_pipeline(
        repository,
        [article(index, text.replace("word", f"article{index}_word")) for index in range(3)],
        config,
    )
    counts = manifest["review_diversity"]["chunks_per_article"]
    assert len(counts) == 3
    assert max(counts.values()) == 2
    assert manifest["accepted_chunks"] == 6


def test_structured_reference_section_is_excluded() -> None:
    result = chunk_document_detailed(
        "Useful factual introduction with complete prose.\n\nReferences\n\nSmith, J. (2020). Book.",
        WordTokenizer(), target_tokens=20, maximum_tokens=30,
        minimum_tokens=3, overlap_tokens=1, reference_section_behavior="exclude",
    )
    assert result.excluded_reference_sections == 1
    assert all("Smith" not in chunk.text for chunk in result.chunks)


def test_embedded_reference_heading_is_excluded() -> None:
    result = chunk_document_detailed(
        "Useful factual introduction with complete prose.\nReferences\nSmith, J. (2020). Book.",
        WordTokenizer(), target_tokens=20, maximum_tokens=30,
        minimum_tokens=3, overlap_tokens=1, reference_section_behavior="exclude",
    )
    assert result.excluded_reference_sections == 1
    assert [chunk.text for chunk in result.chunks] == [
        "Useful factual introduction with complete prose."
    ]


@pytest.mark.parametrize(
    "heading", ["REFERENCES", "  General-and-cited references: ", "Further Reading!!!"]
)
def test_reference_heading_matching_is_case_and_punctuation_insensitive(
    heading: str,
) -> None:
    assert is_reference_section_heading(heading)


def test_legitimate_sources_word_in_prose_is_retained() -> None:
    text = "Scientists compare energy sources in ordinary explanatory prose."
    chunks = chunk_document(
        text, WordTokenizer(), target_tokens=20, maximum_tokens=30,
        minimum_tokens=3, overlap_tokens=1, reference_section_behavior="exclude",
    )
    assert [chunk.text for chunk in chunks] == [text]


def test_minimum_token_threshold_is_enforced() -> None:
    result = chunk_document_detailed(
        "Tiny fragment.", WordTokenizer(), target_tokens=8, maximum_tokens=10,
        minimum_tokens=3, overlap_tokens=0,
    )
    assert not result.chunks
    assert result.rejection_counts["below_minimum_chunk_tokens"] == 1


def test_tiny_trailing_chunk_is_rejected_when_merge_is_unsafe() -> None:
    result = chunk_document_detailed(
        "one two three four five six seven eight.\n\nAppendix\n\nsmall bit.",
        WordTokenizer(), target_tokens=8, maximum_tokens=8,
        minimum_tokens=3, overlap_tokens=0,
    )
    assert [chunk.token_count for chunk in result.chunks] == [8]
    assert result.rejection_counts["below_minimum_chunk_tokens"] == 1


def _final_quality(text: str, token_count: int = 20):
    return assess_final_chunk(
        text,
        token_count=token_count,
        minimum_tokens=3,
        maximum_list_like_line_ratio=0.75,
        minimum_prose_sentences_for_list_chunk=2,
        boundary_start_type="paragraph",
        boundary_end_type="paragraph",
        training_mode=True,
    )


def test_punctuation_only_chunk_is_rejected() -> None:
    assert _final_quality("--- !!! ???", 3).rejection_reason == "low_information"


def test_heading_only_chunk_is_rejected() -> None:
    assert _final_quality("Important Historical Notes", 3).rejection_reason == "low_information"


def test_list_introduction_without_list_is_rejected() -> None:
    result = _final_quality("The principal examples are as follows:", 6)
    assert result.rejection_reason == "low_information"


def test_chunks_never_begin_or_end_inside_alphanumeric_words() -> None:
    text = (
        "AlphaLongWord BetaLongWord GammaLongWord DeltaLongWord "
        "EpsilonLongWord ZetaLongWord."
    )
    chunks = chunk_document(
        text, CharacterTokenizer(), target_tokens=28, maximum_tokens=32,
        minimum_tokens=8, overlap_tokens=4,
    )
    source_words = {word.rstrip(".") for word in text.split()}
    assert chunks
    for chunk in chunks:
        assert chunk.text.split()[0].rstrip(".") in source_words
        assert chunk.text.split()[-1].rstrip(".") in source_words
        assert chunk.boundary_start_type != "token_fallback"
        assert chunk.boundary_end_type != "token_fallback"


def test_sentence_boundary_is_preferred() -> None:
    text = (
        "First sentence contains several useful words for context. "
        "Second sentence also contains several useful words for context. "
        "Third sentence remains complete and coherent."
    )
    chunks = chunk_document(
        text, WordTokenizer(), target_tokens=10, maximum_tokens=12,
        minimum_tokens=3, overlap_tokens=2,
    )
    assert chunks[1].text.startswith("Second sentence")
    assert chunks[1].boundary_start_type == "sentence"


def test_date_list_dominated_chunk_is_rejected() -> None:
    text = "\n".join(
        ["1901 – Alice Example", "1902 – Bob Example", "1903 – Carol Example", "1904 – David Example"]
    )
    assert _final_quality(text).rejection_reason == "list_dominated"


def test_bibliography_dominated_chunk_is_rejected() -> None:
    text = "\n".join(
        [
            "Smith, John (1999). First Book.", "Doe, Jane (2000). Second Book.",
            "Brown, Bob (2001). Third Book.", "Jones, Jim (2002). Fourth Book.",
        ]
    )
    assert _final_quality(text).rejection_reason == "list_dominated"


def test_names_only_list_is_rejected() -> None:
    text = "Alice Example\nBob Example\nCarol Example\nDavid Example"
    assert _final_quality(text).rejection_reason == "list_dominated"


def test_unbulleted_calendar_list_is_rejected() -> None:
    text = "\n".join(
        [
            "World Art Day",
            "Flag Day (Ireland)",
            "National Panchayati Raj Day (India)",
            "Teachers' Day (Paraguay)",
        ]
    )
    assert _final_quality(text).rejection_reason == "list_dominated"


def test_normal_prose_with_small_list_is_retained() -> None:
    text = (
        "The article explains the subject with multiple complete sentences. "
        "It supplies context before a short list.\n- First useful example\n"
        "The conclusion returns to ordinary factual prose."
    )
    assert _final_quality(text).rejection_reason is None


@pytest.mark.parametrize(
    "text",
    [
        "The city is currently about . More context follows.",
        "The total land area of, according to the source.",
        "The result is (). Additional prose follows.",
        "The measured value = . Additional prose follows.",
    ],
)
def test_missing_source_values_are_rejected(text: str) -> None:
    assert _final_quality(text).rejection_reason == "missing_source_value"


WIKIMEDIA_REJECTED_REVIEW_FIXTURES = (
    (
        "624:0007 Alaska",
        "The tanker spilled more than of crude oil over of coastline. "
        "Relief aircraft later delivered supplies to nearby communities.",
        "missing_source_value",
    ),
    (
        "675:0001 Affirming the consequent",
        "The argument uses the consequent, Q, of, to conclude the antecedent. "
        "It can be summarized formally as or, alternatively,.",
        "missing_source_value",
    ),
    (
        "787:0001 Alismatales",
        "The order contains the families listed below.\n"
        "family Alismataceae\nfamily Aponogetonaceae\nfamily Araceae\n"
        "family Butomaceae\nfamily Cymodoceaceae\nfamily Hydrocharitaceae\n"
        "family Juncaginaceae\nfamily Posidoniaceae\nfamily Ruppiaceae",
        "list_dominated",
    ),
    (
        "903:0000 Arable land",
        "Arable land is land used to grow crops. The term often has a more "
        "precise definition:\n\nA later paragraph discusses agricultural statistics.",
        "missing_source_value",
    ),
    (
        "1016:0011 Achill Island",
        "The literature includes the following works.\n"
        "Heinrich Boll: Island Diary, Berlin, 1957\n"
        "Rosa Meehan: The Story of Mayo, Castlebar, 2003\n"
        "James Carney: The Yellow Lady, Dublin, 1986\n"
        "Hugo Hamilton: The Island of Talking, 2007\n"
        "Kevin Barry: Beatlebone, 2015\n"
        "Patricia Byrne: The Veiled Woman of Achill, 2012\n"
        "Mary Murphy: Forgotten Island History, 2011\n"
        "Michael Gallagher: Stick on Stone, 2013",
        "list_dominated",
    ),
    (
        "1097:0008 Armed Forces of Armenia",
        "Military education is provided by several institutions.\n"
        "National Defense Research University\n"
        "Vazgen Sargsyan Military University\n"
        "Monte Melkonian Military Academy\n"
        "Military Academy of Modena\n"
        "Hellenic Military Academy\n"
        "Armenak Khanperyants Military Aviation University\n"
        "Yerevan State Medical University Military Faculty\n"
        "Conscription and Mobilization Service",
        "list_dominated",
    ),
    (
        "1134:0006 Analysis",
        "The term analysis is used in many fields.\n"
        "Policy analysis – evaluation of policy choices\n"
        "Finite element analysis – a simulation technique\n"
        "Link quality analysis – analysis of signal quality\n"
        "Cluster analysis – techniques for finding groups\n"
        "Factor analysis – construction of latent models\n"
        "Regression analysis – study of predictive relationships\n"
        "Sensitivity analysis – study of output variation\n"
        "Spatial analysis – study using geometric properties",
        "list_dominated",
    ),
)


@pytest.mark.parametrize(
    ("fixture_name", "text", "expected_reason"),
    WIKIMEDIA_REJECTED_REVIEW_FIXTURES,
)
def test_completed_review_rejections_are_regression_fixtures(
    fixture_name: str,
    text: str,
    expected_reason: str,
) -> None:
    assert fixture_name
    assert _final_quality(text, len(text.split())).rejection_reason == expected_reason


def test_ordinary_scientific_prose_remains_accepted() -> None:
    text = (
        "Photosynthesis converts light energy into chemical energy in plants. "
        "Chlorophyll absorbs light, and the resulting reactions help produce "
        "sugars while releasing oxygen."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_readable_mathematical_formula_remains_accepted() -> None:
    text = (
        "Modus ponens uses the readable propositions P → Q and P to infer Q. "
        "The expression is complete, and each symbol is explained in the sentence."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_alabama_missing_linguistic_forms_are_rejected() -> None:
    text = (
        "The word for a person of this lineage is (or variously or in different "
        "dialects; the plural form is ). Historical sources use several spellings."
    )
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_empty_plural_form_expression_is_rejected() -> None:
    text = "The singular form remains documented, but the plural form is )."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_repeated_connector_from_removed_term_is_rejected() -> None:
    text = "The term is described in the sources as northern or and southern."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_linguistic_explanation_is_retained() -> None:
    text = (
        "The singular form is Alabamian, while the plural form is Alabamians. "
        "Different dialects preserve several documented pronunciations."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_arithmetic_mean_missing_sample_is_rejected() -> None:
    text = (
        "For example, consider the data sample. The mean and median would normally "
        "be calculated from the listed observations."
    )
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_missing_mathematical_result_is_rejected() -> None:
    text = "For the observations shown above, the mean is, as is the median."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_missing_example_after_such_as_is_rejected() -> None:
    text = (
        "For a sample that cannot be ordered arithmetically, such as, the median "
        "and arithmetic average may differ."
    )
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_mathematical_prose_is_retained() -> None:
    text = (
        "The mean is larger than the median, while the value is unknown for the "
        "unobserved population."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_valid_mathematical_sample_and_formula_are_retained() -> None:
    text = (
        "For example, consider the data sample [1, 2, 3]. Its arithmetic mean is "
        "(1 + 2 + 3) / 3 = 2, and the median is also 2."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_new_missing_expression_rejections_do_not_affect_accounting(
    repository: Path,
) -> None:
    linguistic = (
        "The lineage term is (or variously or in different dialects; the plural "
        "form is )."
    )
    mathematics = (
        "For example, consider the data sample. The mean is, as is the median."
    )
    good = "A clean factual article contains enough complete explanatory prose."
    config = make_config(
        chunking_enabled=False,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    manifest, _, _ = run_pipeline(
        repository,
        [article(1, linguistic), article(2, mathematics), article(3, good)],
        config,
    )
    assert manifest["rejection_reasons"]["missing_source_value"] == 2
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["total_vasu_tokens"] == len(WordTokenizer().encode(good))


WIKIMEDIA_EXTRACTION_REVIEW_FIXTURES = (
    (
        "600:0018 Andorra",
        "According to language statistics released in 2018:\n\n"
        "The official language is Catalan (Catalan: ).",
        "missing_source_value",
    ),
    (
        "624:0025 Alaska",
        "Health insurance\n\n, CVS Health and Premera account for most private "
        "health insurance policies.",
        "malformed_source_text",
    ),
    (
        "736:0023 Albert Einstein",
        "Einstein's \"\" (\"On the Electrodynamics of Moving Bodies\") was "
        "published in 1905.",
        "missing_source_value",
    ),
    (
        "746:0022 Azerbaijan",
        "In 2010 broad-gauge and electrified railways stretched for and "
        "respectively.",
        "missing_source_value",
    ),
    (
        "849:0008 Aircraft",
        "The experimental aircraft flew at Mach 9.68 or on 16 November 2004.",
        "missing_source_value",
    ),
    (
        "909:0006 Anglican Communion",
        "It establishes four principles with these words:\n\n"
        "Instruments of communion\nThe next section discusses administration.",
        "missing_source_value",
    ),
    (
        "1209:0007 Area",
        "The area of the parallelogram is equal to the rectangle:\n"
        "(parallelogram).\nThe geometric discussion then continues.",
        "missing_source_value",
    ),
    (
        "1365:0000 Ammonia",
        "Ammonia is a compound of nitrogen and hydrogen with the formula. "
        "It is a colourless gas.",
        "missing_source_value",
    ),
)


@pytest.mark.parametrize(
    ("fixture_name", "text", "expected_reason"),
    WIKIMEDIA_EXTRACTION_REVIEW_FIXTURES,
)
def test_remaining_extraction_review_failures_are_regression_fixtures(
    fixture_name: str,
    text: str,
    expected_reason: str,
) -> None:
    assert fixture_name
    assert _final_quality(text, len(text.split())).rejection_reason == expected_reason


def test_valid_chemical_formula_is_retained() -> None:
    text = "Ammonia is a compound of nitrogen and hydrogen with the formula NH3."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_valid_area_formula_is_retained() -> None:
    text = (
        "The rectangle has area = base × height, while a triangle has area = "
        "1/2 × base × height."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_valid_translated_language_term_is_retained() -> None:
    text = (
        "The toponymy commission is called Comissió de Toponímia in Catalan."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_valid_converted_measurement_is_retained() -> None:
    text = "The aircraft reached Mach 9.68, approximately 11,850 km/h."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_valid_short_enumeration_after_introduction_is_retained() -> None:
    text = (
        "The following principles:\n\n- Preserve complete source values.\n"
        "- Reject incomplete extracted expressions."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_extraction_failure_rejections_do_not_affect_accounting(
    repository: Path,
) -> None:
    good = "A clean factual article contains enough complete explanatory prose."
    config = make_config(
        chunking_enabled=False,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    rejected = [
        article(index, fixture[1])
        for index, fixture in enumerate(
            WIKIMEDIA_EXTRACTION_REVIEW_FIXTURES,
            start=1,
        )
    ]
    manifest, _, _ = run_pipeline(
        repository,
        [*rejected, article(100, good)],
        config,
    )
    assert manifest["rejection_reasons"]["missing_source_value"] == 7
    assert manifest["rejection_reasons"]["malformed_source_text"] == 1
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["total_vasu_tokens"] == len(WordTokenizer().encode(good))


WIKIMEDIA_FINAL_REVIEW_FIXTURES = (
    (
        "690:0000 Aruba",
        "Aruba (, or, ), officially the Country of Aruba (; ), lies about "
        "north of the mainland. It measures long and across at its widest point. "
        "Its area is and it is densely populated.",
        "missing_source_value",
    ),
    (
        "708:0000 Transport in Angola",
        "Railways:\nLuanda Railway\nBenguela Railway\nMocamedes Railway\n\n"
        "The principal routes are operational.\n\nWaterways:\n"
        "River route\nCanal route\nHarbor route\n\nPipelines:\n"
        "gas 352 km; liquid petroleum gas 85 km; crude oil 1,065 km",
        "list_dominated",
    ),
    (
        "803:0000 Arabic",
        "Arabic (,;, or ) is a Semitic language. One variety is written in "
        "Latin script (in Senegal).; Maltese also uses a Latin script.",
        "missing_source_value",
    ),
    (
        "1016:0007 Achill Island",
        "Thomas's church)\nInnisbiggle Island church\nOther:\n"
        "House of Prayer, Achill\n\nA prose discussion follows the broken list.",
        "malformed_source_text",
    ),
    (
        "1291:0000 Antarctic Treaty System",
        "The countries cooperated during the scientific program., the treaty "
        "has 56 parties.",
        "missing_source_value",
    ),
    (
        "1370:0000 Ambrose",
        "Ambrose of Milan (; 4 April 397) wrote several works, including the "
        "exegetical (386–390).",
        "missing_source_value",
    ),
)


@pytest.mark.parametrize(
    ("fixture_name", "text", "expected_reason"),
    WIKIMEDIA_FINAL_REVIEW_FIXTURES,
)
def test_six_remaining_review_failures_are_regression_fixtures(
    fixture_name: str,
    text: str,
    expected_reason: str,
) -> None:
    assert fixture_name
    assert _final_quality(text, len(text.split())).rejection_reason == expected_reason


def test_missing_geographical_distance_and_area_are_rejected() -> None:
    text = (
        "The island lies about north of the peninsula. It measures long from end "
        "to end and across at its widest point. Its area is and it is populated."
    )
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_truncated_numeric_statistic_is_rejected() -> None:
    text = "Pipelines\ncrude oil 1,\nA later paragraph discusses construction."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "malformed_source_text"
    )


def test_valid_infrastructure_statistic_is_retained() -> None:
    text = "Pipelines carried crude oil for 1,065 km and gas for 352 km in 2013."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_connector_only_pronunciation_parentheses_are_rejected() -> None:
    text = "The language name is written Arabic (,;, or ) in the source."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_malformed_parenthetical_punctuation_is_rejected() -> None:
    text = "The variety is written in Latin script (in Senegal).; Maltese differs."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "malformed_source_text"
    )


def test_list_fragment_chunk_start_is_rejected() -> None:
    text = "Thomas's church)\nInnisbiggle Island church\nOther:\nHouse of Prayer"
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "malformed_source_text"
    )


def test_multiple_structured_list_blocks_are_rejected() -> None:
    text = (
        "Churches:\nNorth Church\nSouth Church\nIsland Church\n\n"
        "A short note separates the groups.\n\nSchools:\n"
        "Harbor School\nVillage School\nCommunity School"
    )
    assert _final_quality(text, len(text.split())).rejection_reason == "list_dominated"


def test_normal_prose_with_short_supporting_list_remains_accepted() -> None:
    text = (
        "The report explains two verified examples in context.\n"
        "- The first example has a complete value.\n"
        "- The second example has a complete value.\n"
        "The concluding sentence explains why both examples matter."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_missing_introductory_date_before_comma_is_rejected() -> None:
    text = "Scientific cooperation was achieved., the treaty has 56 parties."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_dated_treaty_statement_is_retained() -> None:
    text = "As of 2024, the treaty has 56 parties and remains in force."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_missing_birth_date_in_biographical_parenthesis_is_rejected() -> None:
    text = "A historical theologian (; 4 April 397) served as a bishop."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_missing_work_title_before_date_range_is_rejected() -> None:
    text = "The author completed the exegetical (386–390) during this period."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_lifespan_and_work_title_prose_is_retained() -> None:
    text = (
        "Ambrose of Milan (c. 339–4 April 397) wrote the exegetical work "
        "Exposition of the Christian Faith (386–390)."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_final_review_rejections_do_not_affect_accounting(repository: Path) -> None:
    good = "A clean factual article contains enough complete explanatory prose."
    config = make_config(
        chunking_enabled=False,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    rejected = [
        article(index, fixture[1])
        for index, fixture in enumerate(WIKIMEDIA_FINAL_REVIEW_FIXTURES, start=1)
    ]
    manifest, _, _ = run_pipeline(
        repository,
        [*rejected, article(100, good)],
        config,
    )
    assert manifest["rejection_reasons"]["missing_source_value"] == 4
    assert manifest["rejection_reasons"]["malformed_source_text"] == 1
    assert manifest["rejection_reasons"]["list_dominated"] == 1
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["total_vasu_tokens"] == len(WordTokenizer().encode(good))


@pytest.mark.parametrize(
    "text",
    [
        "The international success of allowed the director to continue.",
        "His next project,, another epic, entered production.",
        "After s release the film received attention.",
    ],
)
def test_general_missing_entity_patterns_are_rejected(text: str) -> None:
    assert _final_quality(text).rejection_reason == "malformed_source_text"


def test_quality_rejected_chunks_do_not_affect_accounting(repository: Path) -> None:
    good = "A clean factual article contains enough complete explanatory prose."
    config = make_config(
        chunking_enabled=False, contamination_check_enabled=False,
        exact_deduplication_enabled=False, near_deduplication_enabled=False,
    )
    bad_rows = [
        article(index, fixture[1])
        for index, fixture in enumerate(WIKIMEDIA_REJECTED_REVIEW_FIXTURES, start=1)
    ]
    manifest, _, _ = run_pipeline(
        repository,
        [*bad_rows, article(100, good)],
        config,
    )
    assert manifest["rejection_reasons"]["missing_source_value"] == 3
    assert manifest["rejection_reasons"]["list_dominated"] == 4
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["total_vasu_tokens"] == len(WordTokenizer().encode(good))


def test_smoke_and_production_reference_policies_agree() -> None:
    config = replace(make_config(), reference_section_behavior="exclude")
    assert config.for_smoke_test().reference_section_behavior == "exclude"


def test_broad_review_reference_flagging_remains_available() -> None:
    result = chunk_document_detailed(
        "Useful factual introduction.\n\nReferences\n\nSmith, J. (2020). Book.",
        WordTokenizer(), target_tokens=20, maximum_tokens=30,
        minimum_tokens=3, overlap_tokens=1, reference_section_behavior="flag",
    )
    assert any(chunk.section_title == "References" for chunk in result.chunks)
    assert result.excluded_reference_sections == 0


WIKIMEDIA_EIGHT_REVIEW_FIXTURES = (
    (
        "633:0005 Algae",
        "Symbiotic algae provide photosynthates to their hosts. Examples are:",
        "missing_source_value",
    ),
    (
        "1140:0002 Amplitude modulation",
        "The ITU designated the types of amplitude modulation:",
        "missing_source_value",
    ),
    (
        "1182:0005 Athena",
        'The epithet is derived either from, meaning "to brandish", or from '
        'and related words, meaning "young woman".',
        "missing_source_value",
    ),
    (
        "1210:0005 Astronomical unit",
        "One astronomer gave a mean solar distance of Earth radii, while "
        "another used a mean solar distance of Earth radii.",
        "missing_source_value",
    ),
    (
        "1313:0002 Aromatic compound",
        "An example is a direct arylation of perfluorobenzenes\n\n"
        "Hydrogenation\nHydrogenation creates saturated rings.",
        "missing_source_value",
    ),
    (
        "1335:0001 Associative property",
        "A product of four elements may be written in five possible ways:\n\n"
        "If the operation is associative, every expression has the same result.",
        "missing_source_value",
    ),
    (
        "1370:0019 Ambrose",
        "First work, translated by A. Editor, vol. 1, (Oxford: Example Press, "
        "1998) [Contains translations of three works]\n"
        "Second work, translated by B. Editor, vol. 2, (London: Sample Press, "
        "1999) [Contains translations of four works]\n"
        "Third work, edited by C. Scholar, vol. 3, (Paris: Test Press, 2000) "
        "[Contains the Latin text]\n"
        "Fourth work, translated by D. Scholar, vol. 4, (Dublin: Demo Press, "
        "2001) [Contains commentary and notes]",
        "list_dominated",
    ),
    (
        "1514:0000 Albert, Duke of Prussia",
        "Albert of Prussia (; 17 May 149020 March 1568) was a German prince.",
        "missing_source_value",
    ),
)


@pytest.mark.parametrize(
    ("fixture_name", "text", "expected_reason"),
    WIKIMEDIA_EIGHT_REVIEW_FIXTURES,
)
def test_eight_latest_review_failures_are_regression_fixtures(
    fixture_name: str,
    text: str,
    expected_reason: str,
) -> None:
    assert fixture_name
    assert _final_quality(text, len(text.split())).rejection_reason == expected_reason


def test_missing_enumeration_after_examples_are_is_rejected() -> None:
    text = "The organisms form several symbioses. Examples are:"
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_short_enumeration_after_examples_are_is_retained() -> None:
    text = (
        "The organisms form several well-described symbioses.\n"
        "Examples are:\n- lichens\n- corals\n"
        "Both examples have distinct hosts and complete descriptions."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_missing_classification_after_introductory_colon_is_rejected() -> None:
    text = "The signal types are classified as:"
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_missing_etymological_source_terms_are_rejected() -> None:
    text = 'The term is derived from, meaning "to carry", and from and related words.'
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_non_latin_etymology_is_retained() -> None:
    text = 'The epithet derives from Greek πάλλω, meaning "to brandish a weapon".'
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_missing_numerical_measurement_is_rejected() -> None:
    text = "The estimate gave a distance of Earth radii and a mass of kilograms."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_earth_radii_measurement_is_retained() -> None:
    text = "The estimate gave a mean solar distance of 1,210 Earth radii."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_missing_chemical_reaction_after_colon_is_rejected() -> None:
    text = (
        "The enolate reacts with methyl iodide to form the product:\n\n"
        "Cycloadditions\nOther reactions occur through excimers."
    )
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_prose_only_chemical_reaction_is_retained() -> None:
    text = (
        "The enolate reacts with methyl iodide and forms "
        "2-methyl-1,3-cyclohexanedione as the product."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_missing_mathematical_expression_set_is_rejected() -> None:
    text = (
        "The product may be grouped in five possible ways:\n\n"
        "Every grouping has the same value when the operation is associative."
    )
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_missing_operands_around_equivalence_are_rejected() -> None:
    text = "The operation is associative; thus, is equivalent to, but most commonly means, which is not equivalent."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_valid_associative_law_formulas_are_retained() -> None:
    text = "The associative law states (a × b) × c = a × (b × c)."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_bibliography_dominated_publication_entries_are_rejected() -> None:
    text = WIKIMEDIA_EIGHT_REVIEW_FIXTURES[6][1]
    assert _final_quality(text, len(text.split())).rejection_reason == "list_dominated"


def test_normal_prose_with_a_few_citations_is_retained() -> None:
    text = (
        "The study explains how the manuscripts changed over time. Smith (1998) "
        "described the earliest copy, while Jones (2001) compared a later edition. "
        "Both citations support the historical explanation."
    )
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_editions_and_works_cited_are_reference_section_headings() -> None:
    assert is_reference_section_heading("Editions")
    assert is_reference_section_heading("Publications")
    assert is_reference_section_heading("Works cited")


def test_malformed_joined_lifespan_dates_are_rejected() -> None:
    text = "The ruler (17 May 149020 March 1568) governed for many years."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "malformed_source_text"
    )


@pytest.mark.parametrize(
    "lifespan",
    ["17 May 1490 – 20 March 1568", "17 May 1490–20 March 1568", "1490–1568"],
)
def test_valid_lifespan_dates_are_retained(lifespan: str) -> None:
    text = f"The ruler ({lifespan}) governed the duchy for many years."
    assert _final_quality(text, len(text.split())).rejection_reason is None


def test_empty_pronunciation_field_before_semicolon_is_rejected() -> None:
    text = "The ruler (; 17 May 1490 – 20 March 1568) governed the duchy."
    assert _final_quality(text, len(text.split())).rejection_reason == (
        "missing_source_value"
    )


def test_latest_review_rejections_do_not_affect_accounting(repository: Path) -> None:
    good = "A clean factual article contains enough complete explanatory prose."
    config = make_config(
        chunking_enabled=False,
        contamination_check_enabled=False,
        exact_deduplication_enabled=False,
        near_deduplication_enabled=False,
    )
    rejected = [
        article(index, fixture[1])
        for index, fixture in enumerate(WIKIMEDIA_EIGHT_REVIEW_FIXTURES, start=1)
    ]
    manifest, _, _ = run_pipeline(
        repository,
        [*rejected, article(100, good)],
        config,
    )
    assert manifest["rejection_reasons"]["missing_source_value"] == 7
    assert manifest["rejection_reasons"]["list_dominated"] == 1
    assert manifest["accepted_parent_documents"] == 1
    assert manifest["accepted_chunks"] == 1
    assert manifest["total_vasu_tokens"] == len(WordTokenizer().encode(good))
