"""Emit the exact clean-runtime eligibility report without writing artifacts."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.smoke_vasu_140m_authorization_protocol_postcommit import (  # noqa: E402
    GATE_SHA256,
    IMPLEMENTATION_COMMIT,
    SMOKE_SHA256,
    TEST_SHA256,
    main as emit_postcommit_report,
)
from vasu.data.vasu_140m_authorization_protocol import (  # noqa: E402
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


def probe_repository(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def build_runtime_report(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    repository_probe: Callable[[Path], tuple[str, bool]] = probe_repository,
) -> dict[str, object]:
    """Build a report only for an exact clean descendant of accepted code."""

    root = repository_root.resolve()
    runtime_commit, clean = repository_probe(root)
    if not clean:
        raise ValueError("runtime eligibility requires a clean worktree and index")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, runtime_commit],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("runtime commit does not descend from accepted implementation")
    identities = {
        "authorization_protocol_sha256": (root / PROTOCOL_PATH, GATE_SHA256),
        "authorization_tests_sha256": (
            root / "tests/test_vasu_140m_authorization_protocol.py",
            TEST_SHA256,
        ),
        "authorization_smoke_sha256": (
            root / "scripts/smoke_vasu_140m_authorization_protocol.py",
            SMOKE_SHA256,
        ),
        "production_builder_sha256": (
            root / IMPLEMENTATION_PATH,
            IMPLEMENTATION_SHA256,
        ),
    }
    observed_identities: dict[str, str] = {}
    for field, (path, expected) in identities.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"runtime identity mismatch: {field}")
        observed_identities[field] = observed
    output = StringIO()
    with redirect_stdout(output):
        emit_postcommit_report()
    accepted_report = json.loads(output.getvalue())
    protected = [
        PRODUCTION_RELEASE_PATH,
        PRODUCTION_MANIFEST_PATH,
        RECEIPT_DIRECTORY,
    ]
    if any((root / path).exists() for path in protected):
        raise ValueError("a protected production path exists")
    report: dict[str, object] = {
        "schema_id": "vasu.production-release-runtime-eligibility.v1",
        "runtime_commit": runtime_commit,
        "worktree_and_index_clean": True,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        **observed_identities,
        "accepted_postcommit_qualification_sha256": accepted_report[
            "qualification_sha256"
        ],
        "assignment_sha256": accepted_report["assignment_sha256"],
        "protected_paths_absent": protected,
        "authorization_envelope_created": False,
        "production_release_created": False,
        "publication_authorized": False,
        "training_authorized": False,
        "review_decision_storage": "detached_from_repository",
    }
    report["runtime_eligibility_sha256"] = sha256_json(report)
    return report


def main() -> None:
    print(json.dumps(build_runtime_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
