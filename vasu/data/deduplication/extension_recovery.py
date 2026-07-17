"""Read-only audit helpers for historical FineWeb extension provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3


RECOVERY_CLASSIFICATIONS = {
    "exact_reconstruction_possible",
    "source_id_reacquisition_possible",
    "approximate_reconstruction_only",
    "unrecoverable",
}


@dataclass(frozen=True)
class ExtensionRecoveryAudit:
    classification: str
    repository: str | None
    dataset_config: str | None
    revision: str | None
    split: str | None
    extension_fingerprints: int
    retained_source_ids: int
    missing_source_ids: int


def historical_extension_fingerprint(text: str) -> str:
    """Reproduce the recorded v1 extension exact-deduplication hash."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def historical_hash_matches(text: str, expected_sha256: str) -> bool:
    return historical_extension_fingerprint(text) == expected_sha256.casefold()


def audit_extension_recovery(metadata_path: Path, database_path: Path) -> ExtensionRecoveryAudit:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    with sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(fingerprints)")
        }
        if not {"fingerprint", "origin", "source_id"}.issubset(columns):
            raise ValueError("historical extension database schema is incompatible")
        total, retained = connection.execute(
            "SELECT COUNT(*), COUNT(source_id) FROM fingerprints WHERE origin='extension'"
        ).fetchone()
    total = int(total)
    retained = int(retained)
    missing = total - retained
    evidence_complete = all(
        metadata.get(field)
        for field in ("dataset_repository", "dataset_config", "dataset_revision", "split")
    )
    if evidence_complete and total > 0 and retained == total:
        classification = "source_id_reacquisition_possible"
    elif evidence_complete and total > 0:
        classification = "approximate_reconstruction_only"
    else:
        classification = "unrecoverable"
    return ExtensionRecoveryAudit(
        classification=classification,
        repository=metadata.get("dataset_repository"),
        dataset_config=metadata.get("dataset_config"),
        revision=metadata.get("dataset_revision"),
        split=metadata.get("split"),
        extension_fingerprints=total,
        retained_source_ids=retained,
        missing_source_ids=missing,
    )
