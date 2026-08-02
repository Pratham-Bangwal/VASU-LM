from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from vasu.data.vasu_140m_base_records import compile_base_text_chunk
from vasu.data.vasu_140m_base_release import (
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_RELEASE_PATH,
    SOURCE_EVIDENCE_SCHEMA_ID,
    SPEC_SCHEMA_ID,
    QualifiedBaseChunk,
    qualify_base_release,
    source_evidence_identity,
    specification_identity,
    validate_qualification_report,
    validate_release_specification,
    validate_release_specification_files,
)
from vasu.data.vasu_140m_records import (
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    SPECIFICATION_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


class ReversibleTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) + 10 for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token - 10) for token in token_ids)


def _binding(root: Path, relative: str) -> dict[str, str]:
    path = root / relative
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fixture_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative, source in (
        ("assets/tokenizer.json", ROOT / "assets/tokenizer.json"),
        (
            "vasu/data/vasu_140m_base_records.py",
            ROOT / "vasu/data/vasu_140m_base_records.py",
        ),
        (
            "vasu/data/vasu_140m_base_release.py",
            ROOT / "vasu/data/vasu_140m_base_release.py",
        ),
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    statuses = {
        "admission": "approved",
        "acquisition_receipt": "completed",
        "raw_inventory": "complete",
        "normalization_manifest": "complete",
        "quarantine_decision": "accepted",
        "deduplication_evidence": "complete",
        "split_assignment": "complete",
        "selection_index": "complete",
    }
    for source_id in ("source-a", "source-b"):
        for kind in (
            "admission",
            "acquisition_receipt",
            "raw_inventory",
            "normalization_manifest",
            "quarantine_decision",
            "deduplication_evidence",
            "split_assignment",
            "selection_index",
        ):
            subject_path = root / f"subjects/{source_id}/{kind}.json"
            subject_path.parent.mkdir(parents=True, exist_ok=True)
            subject_path.write_text(
                f'{{"fixture_only":true,"kind":"{kind}","source_id":"{source_id}"}}\n',
                encoding="utf-8",
            )
            decision = None
            if kind in {"admission", "quarantine_decision"}:
                decision_path = root / f"decisions/{source_id}/{kind}.json"
                decision_path.parent.mkdir(parents=True, exist_ok=True)
                decision_path.write_text(
                    f'{{"accepted":true,"kind":"{kind}","source_id":"{source_id}"}}\n',
                    encoding="utf-8",
                )
                decision = _binding(root, decision_path.relative_to(root).as_posix())
            envelope: dict[str, object] = {
                "schema_id": SOURCE_EVIDENCE_SCHEMA_ID,
                "kind": kind,
                "source_id": source_id,
                "source_revision": "revision-1",
                "subject": _binding(root, subject_path.relative_to(root).as_posix()),
                "decision": decision,
                "status": statuses[kind],
                "complete": True,
                "fixture_only": True,
                "additional_acquisition_authorized": False,
                "publication_authorized": False,
                "training_authorized": False,
                "evidence_sha256": "0" * 64,
            }
            envelope["evidence_sha256"] = source_evidence_identity(envelope)
            path = root / f"evidence/{source_id}/{kind}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(envelope, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    for name in ("development", "held-out"):
        path = root / f"evaluation/inventories/{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"fixture_only":true,"name":"{name}"}}\n', encoding="utf-8")
    return root


def _chunks(
    tokenizer: ReversibleTokenizer,
) -> dict[tuple[str, str], list[QualifiedBaseChunk]]:
    texts = {
        ("source-a", "train"): ["alpha train one", "alpha train two"],
        ("source-a", "development"): ["alpha development"],
        ("source-a", "evaluation"): ["alpha evaluation"],
        ("source-b", "train"): ["beta train one", "beta train two"],
        ("source-b", "development"): ["beta development"],
        ("source-b", "evaluation"): ["beta evaluation"],
    }
    result: dict[tuple[str, str], list[QualifiedBaseChunk]] = {}
    for (source_id, split), values in texts.items():
        rows = []
        for index, text in enumerate(values):
            document_id = f"{source_id}-{split}-document-{index}"
            compiled = compile_base_text_chunk(
                tokenizer=tokenizer,
                source_id=source_id,
                source_revision="revision-1",
                document_id=document_id,
                document_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
                transformation_id="fixture-normalization-v1",
                chunk_id=f"{source_id}-{split}-chunk-{index}",
                chunk_index=0,
                split=split,
                text=text,
            )
            rows.append(QualifiedBaseChunk(compiled, text))
        result[(source_id, split)] = rows
    return result


def _specification(
    root: Path, chunks: dict[tuple[str, str], list[QualifiedBaseChunk]]
) -> dict[str, object]:
    sources = []
    for source_id in ("source-a", "source-b"):
        expected_counts = {}
        for split in ("train", "development", "evaluation"):
            rows = chunks[(source_id, split)]
            expected_counts[split] = {
                "chunk_count": len(rows),
                "real_token_count": sum(len(row.chunk.token_ids) for row in rows),
                "supervised_target_count": sum(
                    sum(row.chunk.stored_mask) for row in rows
                ),
            }
        sources.append(
            {
                "source_id": source_id,
                "source_revision": "revision-1",
                "evidence": {
                    kind: _binding(root, f"evidence/{source_id}/{kind}.json")
                    for kind in (
                        "admission",
                        "acquisition_receipt",
                        "raw_inventory",
                        "normalization_manifest",
                        "quarantine_decision",
                        "deduplication_evidence",
                        "split_assignment",
                        "selection_index",
                    )
                },
                "expected_counts": expected_counts,
                "no_replacement": True,
            }
        )
    value: dict[str, object] = {
        "schema_id": SPEC_SCHEMA_ID,
        "release_id": "vasu-140m-base-release-fixture-v1",
        "qualification_scope": "fixture",
        "repository_commit": COMMIT,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer": _binding(root, "assets/tokenizer.json"),
        "record_specification_sha256": SPECIFICATION_SHA256,
        "record_implementation": _binding(root, "vasu/data/vasu_140m_base_records.py"),
        "builder_implementation": _binding(root, "vasu/data/vasu_140m_base_release.py"),
        "release_directory": PRODUCTION_RELEASE_PATH,
        "external_manifest_path": PRODUCTION_MANIFEST_PATH,
        "source_order": ["source-a", "source-b"],
        "sources": sources,
        "evaluation_inventories": [
            _binding(root, "evaluation/inventories/development.json"),
            _binding(root, "evaluation/inventories/held-out.json"),
        ],
        "minimum_free_bytes": 10_000_000_000,
        "publication_authorized": False,
        "training_authorized": False,
        "specification_sha256": "0" * 64,
    }
    value["specification_sha256"] = specification_identity(value)
    return value


def _fixture(tmp_path: Path):
    tokenizer = ReversibleTokenizer()
    chunks = _chunks(tokenizer)
    root = _fixture_repository(tmp_path)
    specification = _specification(root, chunks)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def factory(source_id: str, split: str):
        return iter(chunks[(source_id, split)])

    return root, scratch, tokenizer, chunks, specification, factory


def _rehash(value: dict[str, object]) -> None:
    value["specification_sha256"] = specification_identity(value)


def test_two_pass_qualification_is_exact_clean_and_non_authorizing(
    tmp_path: Path,
) -> None:
    root, scratch, tokenizer, _, specification, factory = _fixture(tmp_path)
    report = qualify_base_release(
        specification=specification,
        repository_root=root,
        runtime_commit=COMMIT,
        tokenizer=tokenizer,
        stream_factory=factory,
        scratch_parent=scratch,
        free_bytes=lambda _: 100_000_000_000,
    )
    validate_qualification_report(report)
    assert report["passes"][0] == report["passes"][1]
    assert report["qualification_scope"] == "fixture"
    assert report["production_input_evidence_validated"] is False
    assert report["production_release_created"] is False
    assert report["publication_authorized"] is False
    assert report["training_authorized"] is False
    assert list(scratch.iterdir()) == []
    assert not (root / PRODUCTION_RELEASE_PATH).exists()
    assert not (root / PRODUCTION_MANIFEST_PATH).exists()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("family_id", "vasu_60m_v1", "family mismatch"),
        ("publication_authorized", True, "must be false"),
        ("training_authorized", True, "must be false"),
        ("release_directory", "data/processed/other", "directory mismatch"),
        ("minimum_free_bytes", 1, "at least 10 GB"),
    ],
)
def test_frozen_identity_and_authority_fail_closed(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    root, _, _, _, specification, _ = _fixture(tmp_path)
    specification[field] = replacement
    _rehash(specification)
    with pytest.raises(ValueError, match=message):
        validate_release_specification(specification)
    assert not (root / PRODUCTION_RELEASE_PATH).exists()


def test_source_order_and_evidence_paths_are_exact(tmp_path: Path) -> None:
    _, _, _, _, specification, _ = _fixture(tmp_path)
    specification["source_order"].reverse()
    _rehash(specification)
    with pytest.raises(ValueError, match="source_order"):
        validate_release_specification(specification)

    _, _, _, _, specification, _ = _fixture(tmp_path / "second")
    specification["sources"][1]["evidence"]["admission"] = specification["sources"][0][
        "evidence"
    ]["admission"]
    _rehash(specification)
    with pytest.raises(ValueError, match="globally unique"):
        validate_release_specification(specification)


def test_runtime_file_hash_and_production_absence_are_enforced(tmp_path: Path) -> None:
    root, _, _, _, specification, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="runtime commit"):
        validate_release_specification_files(
            specification, root, runtime_commit="b" * 40
        )
    specification["sources"][0]["evidence"]["admission"]["sha256"] = "0" * 64
    _rehash(specification)
    with pytest.raises(ValueError, match="file identity mismatch"):
        validate_release_specification_files(specification, root, runtime_commit=COMMIT)

    root, _, _, _, specification, _ = _fixture(tmp_path / "existing")
    (root / PRODUCTION_RELEASE_PATH).mkdir(parents=True)
    with pytest.raises(ValueError, match="must be absent"):
        validate_release_specification_files(specification, root, runtime_commit=COMMIT)


def test_bound_source_evidence_must_be_semantically_complete(tmp_path: Path) -> None:
    root, _, _, _, specification, _ = _fixture(tmp_path)
    relative = "evidence/source-a/admission.json"
    path = root / relative
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["status"] = "pending"
    envelope["evidence_sha256"] = source_evidence_identity(envelope)
    path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    specification["sources"][0]["evidence"]["admission"] = _binding(root, relative)
    _rehash(specification)
    with pytest.raises(ValueError, match="status mismatch"):
        validate_release_specification_files(specification, root, runtime_commit=COMMIT)

    root, _, _, _, specification, _ = _fixture(tmp_path / "scope")
    specification["qualification_scope"] = "production"
    _rehash(specification)
    with pytest.raises(ValueError, match="fixture_only mismatch"):
        validate_release_specification_files(specification, root, runtime_commit=COMMIT)


def test_count_revision_and_wrong_stream_identity_fail_closed(tmp_path: Path) -> None:
    root, scratch, tokenizer, chunks, specification, factory = _fixture(tmp_path)
    specification["sources"][0]["expected_counts"]["train"]["chunk_count"] += 1
    _rehash(specification)
    with pytest.raises(ValueError, match="boundary masking"):
        validate_release_specification(specification)

    root, scratch, tokenizer, chunks, specification, _ = _fixture(
        tmp_path / "wrong-source"
    )

    def wrong_factory(source_id: str, split: str):
        if source_id == "source-a" and split == "train":
            return iter(chunks[("source-b", "train")])
        return iter(chunks[(source_id, split)])

    with pytest.raises(ValueError, match="wrong source or split"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=wrong_factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )


def test_round_trip_and_text_identity_are_complete(tmp_path: Path) -> None:
    root, scratch, tokenizer, chunks, specification, _ = _fixture(tmp_path)
    original = chunks[("source-a", "train")][0]
    chunks[("source-a", "train")][0] = QualifiedBaseChunk(
        original.chunk, "mutated text"
    )

    def factory(source_id: str, split: str):
        return iter(chunks[(source_id, split)])

    with pytest.raises(ValueError, match="normalized text"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )


def test_cross_source_duplicates_and_parent_split_leakage_fail(tmp_path: Path) -> None:
    root, scratch, tokenizer, chunks, specification, _ = _fixture(tmp_path)
    duplicate = chunks[("source-a", "development")][0]
    chunks[("source-b", "evaluation")][0] = QualifiedBaseChunk(
        duplicate.chunk.__class__(
            **{
                **duplicate.chunk.__dict__,
                "source_id": "source-b",
                "chunk_id": "source-b-evaluation-duplicate",
                "split": "evaluation",
            }
        ),
        duplicate.normalized_text,
    )

    def factory(source_id: str, split: str):
        return iter(chunks[(source_id, split)])

    with pytest.raises(ValueError, match="duplicate chunk text"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )


def test_disk_exhaustion_fails_before_scratch_write(tmp_path: Path) -> None:
    root, scratch, tokenizer, _, specification, factory = _fixture(tmp_path)
    with pytest.raises(OSError, match="insufficient free disk"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 0,
        )
    assert list(scratch.iterdir()) == []


def test_stale_or_linked_scratch_fails_before_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, tokenizer, _, specification, factory = _fixture(tmp_path)
    stale = scratch / ".vasu-140m-base-qualification-pass-1"
    stale.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )
    stale.rmdir()
    monkeypatch.setattr(
        "vasu.data.vasu_140m_base_release._is_link_or_junction",
        lambda path: path == scratch.absolute(),
    )
    with pytest.raises(ValueError, match="canonical directory"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )


def test_fsync_failure_preserves_incomplete_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, scratch, tokenizer, _, specification, factory = _fixture(tmp_path)

    def fail_fsync(_: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr("vasu.data.vasu_140m_base_release.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )
    assert (scratch / ".vasu-140m-base-qualification-pass-1").exists()


def test_post_validation_mutation_is_detected_and_preserved(tmp_path: Path) -> None:
    root, scratch, tokenizer, _, specification, factory = _fixture(tmp_path)

    def mutate(pass_root: Path) -> None:
        path = pass_root / "source-a/train.mask.bin"
        payload = bytearray(path.read_bytes())
        payload[0] = 1
        path.write_bytes(payload)

    with pytest.raises(ValueError, match="artifact identity mismatch"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
            mutation_hook=mutate,
        )
    assert not (scratch / ".vasu-140m-base-qualification-pass-1").exists()
    assert (scratch / ".vasu-140m-base-qualification-pass-2").exists()


def test_non_deterministic_second_pass_is_rejected(tmp_path: Path) -> None:
    root, scratch, tokenizer, chunks, specification, _ = _fixture(tmp_path)
    calls: dict[tuple[str, str], int] = {}

    def factory(source_id: str, split: str):
        key = (source_id, split)
        calls[key] = calls.get(key, 0) + 1
        rows = list(chunks[key])
        if calls[key] == 2 and len(rows) > 1:
            rows.reverse()
        return iter(rows)

    with pytest.raises(ValueError, match="not byte-identical"):
        qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=COMMIT,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
            free_bytes=lambda _: 100_000_000_000,
        )


def test_qualification_report_tampering_fails(tmp_path: Path) -> None:
    root, scratch, tokenizer, _, specification, factory = _fixture(tmp_path)
    report = qualify_base_release(
        specification=specification,
        repository_root=root,
        runtime_commit=COMMIT,
        tokenizer=tokenizer,
        stream_factory=factory,
        scratch_parent=scratch,
        free_bytes=lambda _: 100_000_000_000,
    )
    changed = deepcopy(report)
    changed["training_authorized"] = True
    with pytest.raises(ValueError, match="must be false"):
        validate_qualification_report(changed)
    changed = deepcopy(report)
    changed["passes"][1]["pass_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="non-identical"):
        validate_qualification_report(changed)


def test_frozen_smoke_qualification_reproduces_exactly() -> None:
    from scripts.smoke_vasu_140m_base_release_qualification import run

    expected = json.loads(
        (
            ROOT
            / "evaluation/fixtures/vasu_140m_base_release_qualification_fixture_v1.json"
        ).read_text(encoding="utf-8")
    )
    assert run(repository_commit=expected["repository_commit"]) == expected
