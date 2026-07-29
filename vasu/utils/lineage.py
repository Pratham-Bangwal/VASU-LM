"""Read-only artifact lineage indexing for VASU research evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_files(root: Path) -> Iterable[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def build_lineage_index(repository_root: Path) -> dict[str, Any]:
    """Catalog tracked manifests and evaluation JSON without mutating them."""

    sources = (
        ("manifest", repository_root / "data/manifests"),
        ("evaluation", repository_root / "evaluation/results"),
        ("authorization", repository_root / "configs/authorization"),
    )
    artifacts: list[dict[str, Any]] = []
    for category, root in sources:
        if not root.is_dir():
            continue
        for path in _json_files(root):
            relative = path.relative_to(repository_root).as_posix()
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            artifacts.append(
                {
                    "category": category,
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "format_version": payload.get("format_version")
                    if isinstance(payload, dict)
                    else None,
                    "training_authorized": payload.get("training_authorized")
                    if isinstance(payload, dict)
                    else None,
                }
            )
    return {
        "format_version": "vasu_lineage_index_v1",
        "read_only": True,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
