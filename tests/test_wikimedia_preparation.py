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
from vasu.data.preparation.chunking import chunk_document
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
from vasu.data.preparation.quality import assess_text_quality, repair_mojibake
from vasu.data.preparation.progress import load_progress, save_progress
from vasu.data.preparation.reporting import sha256_file
from vasu.data.preparation.reporting import atomic_write_json
from vasu.data.preparation.schemas import (
    PreparationOutputPaths,
    PreparationProgress,
    WikimediaPreparationConfig,
)
from vasu.data.preparation.token_count import count_tokens
from vasu.data.preparation.wikimedia import (
    acquire_pinned_shard,
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
        max_accepted_documents=20,
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
        minimum_chunk_tokens=128,
        chunk_overlap_tokens=32,
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
    ("field", "value"),
    [
        ("max_download_bytes", 1_000_000_001),
        ("max_raw_examples", 10_001),
        ("max_accepted_documents", 2_001),
        ("max_output_tokens", 2_000_001),
        ("max_output_tokens", 0),
    ],
)
def test_invalid_limits_are_rejected(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="limit|positive"):
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


def test_dry_run_performs_no_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.prepare_wikimedia_pilot as cli

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", root)
    monkeypatch.setattr(
        cli,
        "acquire_pinned_shard",
        lambda *args, **kwargs: pytest.fail("dry-run attempted acquisition"),
    )
    assert cli.main(["--config", str(root / "configs/data/preparation/wikimedia_pilot.json"), "--dry-run"]) == 0


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


def test_atomic_progress_write_and_load(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress = PreparationProgress("hash", "source", "revision", "shard")
    save_progress(path, progress)
    assert load_progress(path).configuration_hash == "hash"
    assert not list(tmp_path.glob("*.tmp"))


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
    assert len(records) == manifest["accepted_documents"] == 3
    assert len({json.loads(line)["document_id"] for line in records}) == 3


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


def test_valid_greek_math_and_accents_survive_cleanup() -> None:
    raw = "Enragés measured 0.5 μm at 23° and reported m−2."
    cleaned, metadata, rejection = clean_training_text(raw)
    assert cleaned == raw
    assert rejection is None
    assert not metadata["encoding_repaired"]


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
    assert manifest["format_version"] == "wikimedia_pilot_v2"
    assert manifest["accepted_chunks"] == len(records) >= 3
    for index, record in enumerate(records):
        assert record["format_version"] == "wikimedia_pilot_document_v2"
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
    assert manifest["total_vasu_tokens"] == 6


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
    assert len({json.loads(line)["chunk_id"] for line in records}) == len(records)


def test_previous_v1_output_is_rejected_clearly(repository: Path) -> None:
    _, config, _ = run_pipeline(repository, [article(1)])
    manifest_path = resolve_paths(config, repository)["manifest_json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "wikimedia_pilot_v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="expected wikimedia_pilot_v2"):
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
    assert review.max_accepted_documents == 20
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
    assert config.max_accepted_documents == 50
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
        max_accepted_documents=20,
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
        max_accepted_documents=20,
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
        max_accepted_documents=6,
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
