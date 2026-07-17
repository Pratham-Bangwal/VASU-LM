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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vasu.data.preparation.contamination import (
    check_contamination,
    load_prompt_evidence,
)
from vasu.data.preparation.deduplication import (
    PilotDeduplicator,
    fineweb_document_index_status,
    load_fineweb_exact_hashes,
)
from vasu.data.preparation.filters import (
    comparison_normalize,
    filter_wikimedia_record,
)
from vasu.data.preparation.progress import load_progress, save_progress
from vasu.data.preparation.reporting import sha256_file
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
    validate_preparation_config,
    validate_preparation_output,
    validate_registry_approval,
)


PINNED_REVISION = "e6057dc557255a03c9c3c47ceab0eb44353b1bc5"
SHARD = "20231101.en/train-00000-of-00041.parquet"


class WordTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


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
    assert records[0]["provenance_metadata"]["row_index"] == 0
    assert records[0]["source_revision"] == PINNED_REVISION


def test_atomic_progress_write_and_load(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    progress = PreparationProgress("hash", "source", "revision", "shard")
    save_progress(path, progress)
    assert load_progress(path).configuration_hash == "hash"
    assert not list(tmp_path.glob("*.tmp"))


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


def test_output_validation_detects_tampering(repository: Path) -> None:
    _, config, _ = run_pipeline(repository, [article(1), article(2)])
    assert validate_preparation_output(config, repository_root=repository)["valid"]
    output = resolve_paths(config, repository)["output_jsonl"]
    output.write_text(output.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_preparation_output(config, repository_root=repository)


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
