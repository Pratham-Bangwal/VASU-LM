from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_wikimedia_quarantined_release as release


PRODUCTION_IDS = [
    "751:0002",
    "791:0010",
    "1227:0000",
    "1270:0000",
    "1348:0008",
    "1348:0012",
    "1520:0022",
    "663:0012",
    "663:0015",
    "663:0017",
    "737:0046",
    "951:0018",
]


def _record(chunk_id: str, parent: str, tokens: int, marker: str) -> dict[str, object]:
    return {
        "format_version": "wikimedia_pilot_document_v3",
        "document_id": chunk_id,
        "parent_document_id": parent,
        "chunk_id": chunk_id,
        "chunk_index": int(chunk_id.split(":")[-1]),
        "chunk_count": 3,
        "title": f"Title {parent}",
        "cleaned_text": marker,
        "token_count": tokens,
        "provenance_metadata": {"source_row_index": int(parent), "marker": marker},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
        newline="",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    root: Path,
    records: list[dict[str, object]],
    quarantine_ids: list[str],
    *,
    source_hash: str | None = None,
) -> tuple[Path, Path, Path]:
    source = root / "source" / "documents.jsonl"
    output = root / "release" / "documents.jsonl"
    manifest = root / "manifests" / "quarantine.json"
    _write_jsonl(source, records)
    by_id = {str(record["chunk_id"]): record for record in records}
    payload = {
        "format_version": release.FORMAT_VERSION,
        "policy_version": "test_policy_v1",
        "created_at": "2026-07-18T00:00:00+00:00",
        "source_dataset_path": str(source.relative_to(root)),
        "source_dataset_sha256": source_hash or _sha256(source),
        "output_path": str(output.relative_to(root)),
        "release_manifest_directory": "manifests/releases",
        "originating_review_artifacts": ["frozen_review.json"],
        "quarantined_chunks": [
            {
                "chunk_id": chunk_id,
                "title": by_id.get(chunk_id, {}).get("title"),
                "reason": f"review rejection for {chunk_id}",
                "originating_review_artifact": "frozen_review.json",
            }
            for chunk_id in quarantine_ids
        ],
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest, source, output


def _build(
    root: Path,
    records: list[dict[str, object]],
    ids: list[str],
) -> tuple[Path, dict[str, object], Path, Path]:
    manifest, source, output = _write_manifest(root, records, ids)
    release_path, report = release.build_quarantined_release(
        manifest, repository_root=root
    )
    return release_path, report, source, output


def test_exact_production_quarantine_ids_are_versioned() -> None:
    payload = json.loads(
        Path("data/manifests/factual/wikimedia_quarantine_ffcbc25f.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["chunk_id"] for item in payload["quarantined_chunks"]] == PRODUCTION_IDS
    assert payload["source_dataset_sha256"] == (
        "ffcbc25f4863f519744212f809ee600bdc7f4a0d5c2d02a0833e1bc4cec6014d"
    )


def test_exact_listed_chunks_removed_and_unlisted_unchanged(tmp_path: Path) -> None:
    records = [
        *[
            _record(chunk_id, chunk_id.split(":")[0], index + 1, chunk_id)
            for index, chunk_id in enumerate(PRODUCTION_IDS)
        ],
        _record("9999:0000", "9999", 99, "retained café provenance"),
    ]
    _, report, _, output = _build(tmp_path, records, PRODUCTION_IDS)
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == records[-1]
    assert report["quarantined_chunk_count"] == 12


def test_order_and_chunk_ids_are_preserved_without_renumbering(tmp_path: Path) -> None:
    records = [
        _record("1:0000", "1", 2, "first"),
        _record("1:0001", "1", 3, "remove"),
        _record("1:0002", "1", 4, "third"),
    ]
    _, _, _, output = _build(tmp_path, records, ["1:0001"])
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert [item["chunk_id"] for item in result] == ["1:0000", "1:0002"]
    assert result == [records[0], records[2]]


def test_input_sha_mismatch_fails_without_output(tmp_path: Path) -> None:
    records = [_record("1:0000", "1", 2, "remove")]
    manifest, _, output = _write_manifest(
        tmp_path, records, ["1:0000"], source_hash="0" * 64
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.build_quarantined_release(manifest, repository_root=tmp_path)
    assert not output.exists()


def test_missing_quarantine_id_fails_without_publication(tmp_path: Path) -> None:
    manifest, _, output = _write_manifest(
        tmp_path, [_record("1:0000", "1", 2, "keep")], ["2:0000"]
    )
    with pytest.raises(ValueError, match="missing from source"):
        release.build_quarantined_release(manifest, repository_root=tmp_path)
    assert not output.exists()
    assert not output.with_name("documents.jsonl.tmp").exists()


def test_duplicate_quarantine_id_fails(tmp_path: Path) -> None:
    manifest, _, _ = _write_manifest(
        tmp_path,
        [_record("1:0000", "1", 2, "remove")],
        ["1:0000", "1:0000"],
    )
    with pytest.raises(ValueError, match="duplicate quarantine chunk IDs"):
        release.load_quarantine_spec(manifest, tmp_path)


def test_duplicate_source_chunk_id_fails(tmp_path: Path) -> None:
    records = [
        _record("1:0000", "1", 2, "first"),
        _record("1:0000", "1", 3, "duplicate"),
        _record("2:0000", "2", 4, "remove"),
    ]
    manifest, _, output = _write_manifest(tmp_path, records, ["2:0000"])
    with pytest.raises(ValueError, match="duplicate source chunk ID"):
        release.build_quarantined_release(manifest, repository_root=tmp_path)
    assert not output.exists()


def test_output_is_atomically_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        _record("1:0000", "1", 2, "keep"),
        _record("1:0001", "1", 3, "remove"),
    ]
    manifest, _, output = _write_manifest(tmp_path, records, ["1:0001"])
    real_replace = release.os.replace
    calls: list[tuple[Path, Path]] = []

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(release.os, "replace", tracked_replace)
    release.build_quarantined_release(manifest, repository_root=tmp_path)
    assert (output.with_name("documents.jsonl.tmp"), output) in calls
    assert not output.with_name("documents.jsonl.tmp").exists()


def test_parent_and_token_accounting(tmp_path: Path) -> None:
    records = [
        _record("1:0000", "1", 2, "keep sibling"),
        _record("1:0001", "1", 3, "remove sibling"),
        _record("2:0000", "2", 5, "remove only child"),
        _record("3:0000", "3", 7, "keep unaffected"),
    ]
    _, report, _, _ = _build(tmp_path, records, ["1:0001", "2:0000"])
    assert report["affected_parent_count"] == 2
    assert report["parents_fully_removed_count"] == 1
    assert report["parents_fully_removed"] == ["2"]
    assert report["release_parent_count"] == 2
    assert report["release_chunk_count"] == 2
    assert report["quarantined_token_count"] == 8
    assert report["release_token_count"] == 9


def test_provenance_fields_remain_unchanged(tmp_path: Path) -> None:
    retained = _record("1:0000", "1", 2, "retained")
    _, _, _, output = _build(
        tmp_path,
        [retained, _record("1:0001", "1", 3, "remove")],
        ["1:0001"],
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["provenance_metadata"] == retained["provenance_metadata"]
    assert result == retained


def test_repeated_build_is_byte_identical_and_hash_stable(tmp_path: Path) -> None:
    records = [
        _record("1:0000", "1", 2, "keep"),
        _record("1:0001", "1", 3, "remove"),
    ]
    manifest, _, output = _write_manifest(tmp_path, records, ["1:0001"])
    first_path, first = release.build_quarantined_release(
        manifest, repository_root=tmp_path
    )
    first_bytes = output.read_bytes()
    first_manifest = first_path.read_bytes()
    second_path, second = release.build_quarantined_release(
        manifest, repository_root=tmp_path
    )
    assert output.read_bytes() == first_bytes
    assert second_path == first_path
    assert second_path.read_bytes() == first_manifest
    assert second["output_sha256"] == first["output_sha256"]


def test_frozen_source_remains_unchanged(tmp_path: Path) -> None:
    records = [
        _record("1:0000", "1", 2, "keep"),
        _record("1:0001", "1", 3, "remove"),
    ]
    manifest, source, _ = _write_manifest(tmp_path, records, ["1:0001"])
    before = source.read_bytes()
    release.build_quarantined_release(manifest, repository_root=tmp_path)
    assert source.read_bytes() == before
    assert _sha256(source) == hashlib.sha256(before).hexdigest()


def test_builder_does_not_create_manual_review_artifacts(tmp_path: Path) -> None:
    records = [
        _record("1:0000", "1", 2, "keep"),
        _record("1:0001", "1", 3, "remove"),
    ]
    manifest, _, _ = _write_manifest(tmp_path, records, ["1:0001"])
    before = set(tmp_path.rglob("*review*"))
    release.build_quarantined_release(manifest, repository_root=tmp_path)
    assert set(tmp_path.rglob("*review*")) == before
