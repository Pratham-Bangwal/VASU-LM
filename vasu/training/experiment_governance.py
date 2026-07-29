"""Read-only governance reports for proposed capability experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PACKET_FORMAT = "vasu_capability_review_packet_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _non_authorizing_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_authorized") is not False:
        raise ValueError(f"{label} must explicitly set training_authorized to false")
    return payload


def build_review_packet(
    *,
    experiment_id: str,
    release_manifests: Mapping[str, Path],
    schedule_manifest: Path | None,
    configuration: Path | None,
    evaluation_snapshots: Mapping[str, Path],
) -> dict[str, Any]:
    """Return deterministic review evidence without modifying any artifact.

    Inputs are deliberately explicit. A packet is not an authorization record:
    it cannot mark an experiment approved or create a launchable configuration.
    """

    if not experiment_id or not release_manifests or not evaluation_snapshots:
        raise ValueError(
            "experiment ID, release manifests, and evaluation snapshots are required"
        )
    releases = {}
    for name, path in sorted(release_manifests.items()):
        _non_authorizing_json(path, label=f"release {name}")
        releases[name] = _identity(path)
    evaluations = {
        name: _identity(path) for name, path in sorted(evaluation_snapshots.items())
    }
    schedule = None
    if schedule_manifest is not None:
        _non_authorizing_json(schedule_manifest, label="schedule manifest")
        schedule = _identity(schedule_manifest)
    config = None
    if configuration is not None:
        _non_authorizing_json(configuration, label="configuration")
        config = _identity(configuration)
    return {
        "format_version": PACKET_FORMAT,
        "experiment_id": experiment_id,
        "review_status": "evidence_collected_not_authorized",
        "training_authorized": False,
        "release_manifests": releases,
        "schedule_manifest": schedule,
        "configuration": config,
        "evaluation_snapshots": evaluations,
        "required_next_decision": "independent_review_then_separate_authorization",
    }
