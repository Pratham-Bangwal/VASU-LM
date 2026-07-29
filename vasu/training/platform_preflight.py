"""Non-authorizing configuration integrity and safety preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vasu.utils.lineage import sha256_file


def preflight_configuration(path: Path) -> dict[str, Any]:
    """Validate a proposed config without authorizing or executing it."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_authorized") is not False:
        raise ValueError("platform preflight accepts only training_authorized=false")
    identities: dict[str, dict[str, str]] = {}
    for name, field, hash_field in (
        (
            "resolved_manifest",
            "resolved_mixture_manifest",
            "resolved_mixture_manifest_sha256",
        ),
        ("parent_checkpoint", "parent_checkpoint", None),
    ):
        value = payload.get(field)
        if isinstance(value, dict):
            value, expected = value.get("path"), value.get("sha256")
        else:
            expected = payload.get(hash_field) if hash_field else None
        if value:
            artifact = Path(value)
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            actual = sha256_file(artifact)
            if expected and expected != actual:
                raise ValueError(f"{name} hash mismatch")
            identities[name] = {"path": artifact.as_posix(), "sha256": actual}
    gates = payload.get("technical_gates", {})
    required = ("replay_safety", "cuda_smoke", "cuda_exact_resume")
    return {
        "format_version": "vasu_platform_preflight_v1",
        "configuration": {"path": path.as_posix(), "sha256": sha256_file(path)},
        "training_authorized": False,
        "identities": identities,
        "technical_gates": {name: gates.get(name, "not_declared") for name in required},
        "readiness": "review_only_not_launchable",
    }
