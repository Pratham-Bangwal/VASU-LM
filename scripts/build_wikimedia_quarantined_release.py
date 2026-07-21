"""Build a deterministic Wikimedia release by quarantining reviewed chunks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from vasu.data.preparation.reporting import atomic_write_json, sha256_file


FORMAT_VERSION = "wikimedia_quarantine_release_v1"
RELEASE_FORMAT_VERSION = "wikimedia_quarantined_release_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class QuarantineEntry:
    chunk_id: str
    title: str | None
    reason: str
    originating_review_artifact: str


@dataclass(frozen=True)
class QuarantineSpec:
    manifest_path: Path
    policy_version: str
    created_at: str
    source_path: Path
    source_sha256: str
    output_path: Path
    release_manifest_directory: Path
    quality_config_path: Path | None
    review_artifacts: tuple[str, ...]
    entries: tuple[QuarantineEntry, ...]


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_quarantine_spec(manifest_path: Path, repository_root: Path) -> QuarantineSpec:
    """Load and strictly validate a quarantine manifest."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"quarantine manifest does not exist: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"quarantine manifest is invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("quarantine manifest root must be an object")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"format_version must be {FORMAT_VERSION!r}")

    source_sha256 = _required_string(payload, "source_dataset_sha256")
    if not SHA256_PATTERN.fullmatch(source_sha256):
        raise ValueError("source_dataset_sha256 must be a lowercase SHA-256 hash")
    raw_entries = payload.get("quarantined_chunks")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("quarantined_chunks must be a non-empty list")

    entries: list[QuarantineEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"quarantined_chunks[{index}] must be an object")
        title = raw.get("title")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ValueError(f"quarantined_chunks[{index}].title must be non-empty or null")
        entries.append(
            QuarantineEntry(
                chunk_id=_required_string(raw, "chunk_id"),
                title=title,
                reason=_required_string(raw, "reason"),
                originating_review_artifact=_required_string(
                    raw, "originating_review_artifact"
                ),
            )
        )
    duplicates = sorted(
        chunk_id
        for chunk_id, count in Counter(entry.chunk_id for entry in entries).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate quarantine chunk IDs: {duplicates}")

    raw_reviews = payload.get("originating_review_artifacts")
    if not isinstance(raw_reviews, list) or not all(
        isinstance(value, str) and value.strip() for value in raw_reviews
    ):
        raise ValueError("originating_review_artifacts must be a list of paths")
    quality_config = payload.get("quality_config_path")
    if quality_config is not None and not isinstance(quality_config, str):
        raise ValueError("quality_config_path must be a path string or null")
    return QuarantineSpec(
        manifest_path=manifest_path,
        policy_version=_required_string(payload, "policy_version"),
        created_at=_required_string(payload, "created_at"),
        source_path=_resolve(
            repository_root, _required_string(payload, "source_dataset_path")
        ),
        source_sha256=source_sha256,
        output_path=_resolve(repository_root, _required_string(payload, "output_path")),
        release_manifest_directory=_resolve(
            repository_root, _required_string(payload, "release_manifest_directory")
        ),
        quality_config_path=(
            _resolve(repository_root, quality_config) if quality_config else None
        ),
        review_artifacts=tuple(raw_reviews),
        entries=tuple(entries),
    )


def _record_identity(record: Mapping[str, Any], line_number: int) -> tuple[str, str, int]:
    chunk_id = record.get("chunk_id")
    parent_id = record.get("parent_document_id")
    token_count = record.get("token_count")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"source line {line_number} has no valid chunk_id")
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError(f"source line {line_number} has no valid parent_document_id")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError(f"source line {line_number} has invalid token_count")
    return chunk_id, parent_id, token_count


def _quality_audit(output_path: Path, quality_config_path: Path | None) -> dict[str, int]:
    if quality_config_path is None:
        return {
            "automatic_reject_findings": 0,
            "manual_review_findings": 0,
            "unexplained_findings": 0,
        }
    # Reuse the existing global audit without changing its quality policy.
    from audit_wikimedia_quality import audit_dataset

    report = audit_dataset(output_path, quality_config_path)
    dispositions = report["disposition_counts"]
    return {
        "automatic_reject_findings": int(dispositions.get("automatic_reject", 0)),
        "manual_review_findings": int(dispositions.get("manual_review", 0)),
        "unexplained_findings": int(report["unexplained_match_count"]),
    }


def validate_release(
    source_path: Path,
    output_path: Path,
    quarantined_ids: set[str],
) -> dict[str, int]:
    """Prove that output is exactly source minus the quarantine set."""
    expected_lines: list[bytes] = []
    expected_ids: list[str] = []
    with source_path.open("rb") as source:
        for line_number, line in enumerate(source, start=1):
            record = json.loads(line.decode("utf-8"))
            chunk_id, _, _ = _record_identity(record, line_number)
            if chunk_id not in quarantined_ids:
                expected_lines.append(line)
                expected_ids.append(chunk_id)
    actual_lines = output_path.read_bytes().splitlines(keepends=True)
    if actual_lines != expected_lines:
        raise ValueError("release bytes differ from source-minus-quarantine transformation")
    actual_ids = [
        str(json.loads(line.decode("utf-8"))["chunk_id"]) for line in actual_lines
    ]
    if actual_ids != expected_ids:
        raise ValueError("release chunk ordering or IDs differ from expected output")
    leaked = sorted(quarantined_ids.intersection(actual_ids))
    if leaked:
        raise ValueError(f"quarantined IDs remain in output: {leaked}")
    return {"validated_chunks": len(actual_ids)}


def build_quarantined_release(
    manifest_path: Path,
    *,
    repository_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build, validate, and describe one deterministic quarantine release."""
    root = (repository_root or Path.cwd()).resolve()
    spec = load_quarantine_spec(manifest_path, root)
    if not spec.source_path.is_file():
        raise ValueError(f"frozen source JSONL does not exist: {spec.source_path}")
    source_hash_before = sha256_file(spec.source_path)
    if source_hash_before != spec.source_sha256:
        raise ValueError(
            "frozen source SHA-256 mismatch: "
            f"expected {spec.source_sha256}, observed {source_hash_before}"
        )

    entries = {entry.chunk_id: entry for entry in spec.entries}
    found: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    input_parents: set[str] = set()
    retained_parents: set[str] = set()
    affected_parents: set[str] = set()
    input_chunks = input_tokens = output_chunks = output_tokens = 0

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = spec.output_path.with_name(spec.output_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with spec.source_path.open("rb") as source, temporary.open("wb") as output:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"invalid source JSONL at line {line_number}: {error}") from error
                chunk_id, parent_id, token_count = _record_identity(record, line_number)
                if chunk_id in seen_ids:
                    raise ValueError(f"duplicate source chunk ID: {chunk_id}")
                seen_ids.add(chunk_id)
                input_parents.add(parent_id)
                input_chunks += 1
                input_tokens += token_count
                entry = entries.get(chunk_id)
                if entry is not None:
                    if entry.title is not None and record.get("title") != entry.title:
                        raise ValueError(
                            f"title mismatch for {chunk_id}: expected {entry.title!r}, "
                            f"observed {record.get('title')!r}"
                        )
                    found[chunk_id] = {
                        "chunk_id": chunk_id,
                        "parent_document_id": parent_id,
                        "title": record.get("title"),
                        "token_count": token_count,
                        "reason": entry.reason,
                        "originating_review_artifact": entry.originating_review_artifact,
                    }
                    affected_parents.add(parent_id)
                    continue
                output.write(line)
                retained_parents.add(parent_id)
                output_chunks += 1
                output_tokens += token_count
            output.flush()
            os.fsync(output.fileno())

        missing = sorted(set(entries).difference(found))
        if missing:
            raise ValueError(f"quarantine chunk IDs missing from source: {missing}")
        if sha256_file(spec.source_path) != source_hash_before:
            raise ValueError("frozen source changed while quarantine release was built")
        os.replace(temporary, spec.output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    validate_release(spec.source_path, spec.output_path, set(entries))
    output_hash = sha256_file(spec.output_path)
    quality = _quality_audit(spec.output_path, spec.quality_config_path)
    parents_fully_removed = affected_parents.difference(retained_parents)
    approved = (
        quality["automatic_reject_findings"] == 0
        and quality["unexplained_findings"] == 0
    )
    release_manifest: dict[str, Any] = {
        "format_version": RELEASE_FORMAT_VERSION,
        "policy_version": spec.policy_version,
        "created_at": spec.created_at,
        "status": "approved" if approved else "blocked",
        "training_authorized": False,
        "source_dataset_path": str(spec.source_path.relative_to(root)),
        "source_dataset_sha256": source_hash_before,
        "output_path": str(spec.output_path.relative_to(root)),
        "output_sha256": output_hash,
        "quarantine_manifest": str(spec.manifest_path.relative_to(root)),
        "originating_review_artifacts": list(spec.review_artifacts),
        "input_parent_count": len(input_parents),
        "input_chunk_count": input_chunks,
        "input_token_count": input_tokens,
        "release_parent_count": len(retained_parents),
        "release_chunk_count": output_chunks,
        "release_token_count": output_tokens,
        "quarantined_chunk_count": len(found),
        "quarantined_token_count": sum(int(item["token_count"]) for item in found.values()),
        "affected_parent_count": len(affected_parents),
        "parents_fully_removed_count": len(parents_fully_removed),
        "parents_fully_removed": sorted(parents_fully_removed),
        "quarantined_chunks": [found[entry.chunk_id] for entry in spec.entries],
        "output_validation": "passed",
        "quality_audit": quality,
        "approval_basis": (
            "Frozen human decisions plus deterministic quarantine, accounting, "
            "byte-preservation, and existing-policy quality validation."
        ),
    }
    release_manifest_path = (
        spec.release_manifest_directory / f"wikimedia_release_{output_hash[:16]}.json"
    )
    atomic_write_json(release_manifest_path, release_manifest)
    return release_manifest_path, release_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/factual/wikimedia_quarantine_ffcbc25f.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    path, report = build_quarantined_release(
        args.manifest.resolve(), repository_root=args.repository_root
    )
    print(json.dumps({**report, "release_manifest_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
