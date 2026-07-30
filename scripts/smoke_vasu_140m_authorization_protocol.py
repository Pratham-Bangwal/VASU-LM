"""Emit read-only qualification evidence for the VASU-140M v2 gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_authorization_protocol import (  # noqa: E402
    ASSIGNMENT_SHA256,
    AUTHORIZATION_V2_SCHEMA_ID,
    IMPLEMENTATION_ANCHOR_COMMIT,
    IMPLEMENTATION_PATH,
    IMPLEMENTATION_SHA256,
    PROTOCOL_PATH,
)
from vasu.data.vasu_140m_production_release import (  # noqa: E402
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_RELEASE_PATH,
    RECEIPT_DIRECTORY,
)
from vasu.data.vasu_140m_records import sha256_json  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    implementation_sha256 = sha256_file(REPOSITORY_ROOT / IMPLEMENTATION_PATH)
    if implementation_sha256 != IMPLEMENTATION_SHA256:
        raise ValueError("accepted production builder identity changed")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_ANCHOR_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
    )
    if ancestry.returncode != 0:
        raise ValueError("accepted production builder anchor is not an ancestor")
    protected = [
        PRODUCTION_RELEASE_PATH,
        PRODUCTION_MANIFEST_PATH,
        RECEIPT_DIRECTORY,
    ]
    if any((REPOSITORY_ROOT / path).exists() for path in protected):
        raise ValueError("a protected production path exists")
    report: dict[str, object] = {
        "schema_id": "vasu.production-release-authorization-qualification.v1",
        "repository_commit": commit,
        "authorization_schema_id": AUTHORIZATION_V2_SCHEMA_ID,
        "implementation_anchor_commit": IMPLEMENTATION_ANCHOR_COMMIT,
        "implementation_sha256": implementation_sha256,
        "authorization_protocol_path": PROTOCOL_PATH,
        "authorization_protocol_sha256": sha256_file(REPOSITORY_ROOT / PROTOCOL_PATH),
        "assignment_sha256": ASSIGNMENT_SHA256,
        "protected_paths_absent": protected,
        "detached_envelope_required": True,
        "authorization_envelope_created": False,
        "production_release_created": False,
        "publication_authorized": False,
        "training_authorized": False,
    }
    report["qualification_sha256"] = sha256_json(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
