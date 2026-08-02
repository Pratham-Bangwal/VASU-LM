"""Run deterministic fixture-only qualification of the base-release builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_base_records import compile_base_text_chunk  # noqa: E402
from vasu.data.vasu_140m_base_release import (  # noqa: E402
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_RELEASE_PATH,
    SOURCE_EVIDENCE_SCHEMA_ID,
    SPEC_SCHEMA_ID,
    QualifiedBaseChunk,
    qualify_base_release,
    source_evidence_identity,
    specification_identity,
)
from vasu.data.vasu_140m_records import (  # noqa: E402
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    SPECIFICATION_SHA256,
)


TEXTS = {
    ("fixture-web", "train"): (
        "A river crosses the quiet valley before reaching the sea.",
        "Careful measurements make scientific results easier to reproduce.",
    ),
    ("fixture-web", "development"): (
        "A stable process records every input identity before execution.",
    ),
    ("fixture-web", "evaluation"): (
        "Deterministic outputs make independent verification possible.",
    ),
    ("fixture-wiki", "train"): (
        "Saturn is the sixth planet from the Sun and has a prominent ring system.",
        "A triangle has three sides and three interior angles.",
    ),
    ("fixture-wiki", "development"): (
        "Liquid water freezes at zero degrees Celsius under standard pressure.",
    ),
    ("fixture-wiki", "evaluation"): (
        "The Pacific Ocean is larger than the Atlantic Ocean.",
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(root: Path, relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": _sha(root / relative)}


def _copy_fixture_dependencies(root: Path) -> None:
    for relative in (
        "assets/tokenizer.json",
        "vasu/data/vasu_140m_base_records.py",
        "vasu/data/vasu_140m_base_release.py",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, target)


def _write_source_evidence(root: Path, source_id: str, kind: str, status: str) -> None:
    subject_relative = f"subjects/{source_id}/{kind}.json"
    subject = root / subject_relative
    subject.parent.mkdir(parents=True, exist_ok=True)
    subject.write_text(
        json.dumps(
            {"fixture_only": True, "kind": kind, "source_id": source_id},
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    decision = None
    if kind in {"admission", "quarantine_decision"}:
        decision_relative = f"decisions/{source_id}/{kind}.json"
        decision_path = root / decision_relative
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(
            json.dumps(
                {"accepted": True, "kind": kind, "source_id": source_id},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        decision = _binding(root, decision_relative)
    envelope: dict[str, object] = {
        "schema_id": SOURCE_EVIDENCE_SCHEMA_ID,
        "kind": kind,
        "source_id": source_id,
        "source_revision": "fixture-revision-1",
        "subject": _binding(root, subject_relative),
        "decision": decision,
        "status": status,
        "complete": True,
        "fixture_only": True,
        "additional_acquisition_authorized": False,
        "publication_authorized": False,
        "training_authorized": False,
        "evidence_sha256": "0" * 64,
    }
    envelope["evidence_sha256"] = source_evidence_identity(envelope)
    envelope_path = root / f"evidence/{source_id}/{kind}.json"
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    envelope_path.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepare_repository(root: Path) -> None:
    _copy_fixture_dependencies(root)
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
    for source_id in ("fixture-web", "fixture-wiki"):
        for kind, status in statuses.items():
            _write_source_evidence(root, source_id, kind, status)
    for name in ("development", "held-out"):
        path = root / f"evaluation/inventories/{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fixture_only": True, "name": name}, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _compile_chunks(
    tokenizer: Tokenizer,
) -> dict[tuple[str, str], tuple[QualifiedBaseChunk, ...]]:
    result = {}
    for (source_id, split), texts in TEXTS.items():
        rows = []
        for index, text in enumerate(texts):
            document_id = f"{source_id}-{split}-document-{index}"
            chunk = compile_base_text_chunk(
                tokenizer=tokenizer,
                source_id=source_id,
                source_revision="fixture-revision-1",
                document_id=document_id,
                document_sha256=hashlib.sha256(document_id.encode()).hexdigest(),
                transformation_id="base-release-smoke-v1",
                chunk_id=f"{source_id}-{split}-chunk-{index}",
                chunk_index=0,
                split=split,
                text=text,
            )
            rows.append(QualifiedBaseChunk(chunk, text))
        result[(source_id, split)] = tuple(rows)
    return result


def _specification(
    root: Path,
    chunks: dict[tuple[str, str], tuple[QualifiedBaseChunk, ...]],
    repository_commit: str,
) -> dict[str, object]:
    kinds = (
        "admission",
        "acquisition_receipt",
        "raw_inventory",
        "normalization_manifest",
        "quarantine_decision",
        "deduplication_evidence",
        "split_assignment",
        "selection_index",
    )
    sources = []
    for source_id in ("fixture-web", "fixture-wiki"):
        counts = {}
        for split in ("train", "development", "evaluation"):
            rows = chunks[(source_id, split)]
            counts[split] = {
                "chunk_count": len(rows),
                "real_token_count": sum(len(item.chunk.token_ids) for item in rows),
                "supervised_target_count": sum(
                    sum(item.chunk.stored_mask) for item in rows
                ),
            }
        sources.append(
            {
                "source_id": source_id,
                "source_revision": "fixture-revision-1",
                "evidence": {
                    kind: _binding(root, f"evidence/{source_id}/{kind}.json")
                    for kind in kinds
                },
                "expected_counts": counts,
                "no_replacement": True,
            }
        )
    value: dict[str, object] = {
        "schema_id": SPEC_SCHEMA_ID,
        "release_id": "vasu-140m-base-release-qualification-fixture-v1",
        "qualification_scope": "fixture",
        "repository_commit": repository_commit,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer": _binding(root, "assets/tokenizer.json"),
        "record_specification_sha256": SPECIFICATION_SHA256,
        "record_implementation": _binding(root, "vasu/data/vasu_140m_base_records.py"),
        "builder_implementation": _binding(root, "vasu/data/vasu_140m_base_release.py"),
        "release_directory": PRODUCTION_RELEASE_PATH,
        "external_manifest_path": PRODUCTION_MANIFEST_PATH,
        "source_order": ["fixture-web", "fixture-wiki"],
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


def run(*, repository_commit: str | None = None) -> dict[str, object]:
    if repository_commit is None:
        repository_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    temp_parent = REPOSITORY_ROOT / "tmp"
    with tempfile.TemporaryDirectory(
        prefix="vasu-140m-base-release-smoke-", dir=temp_parent
    ) as temporary:
        root = Path(temporary) / "repository"
        scratch = Path(temporary) / "scratch"
        root.mkdir()
        scratch.mkdir()
        _prepare_repository(root)
        tokenizer = Tokenizer.from_file(str(root / "assets/tokenizer.json"))
        chunks = _compile_chunks(tokenizer)
        specification = _specification(root, chunks, repository_commit)

        def factory(source_id: str, split: str):
            return iter(chunks[(source_id, split)])

        report = qualify_base_release(
            specification=specification,
            repository_root=root,
            runtime_commit=repository_commit,
            tokenizer=tokenizer,
            stream_factory=factory,
            scratch_parent=scratch,
        )
        if report["qualification_scope"] != "fixture":
            raise RuntimeError("smoke qualification escaped fixture scope")
        return report


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
