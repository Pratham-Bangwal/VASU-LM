"""Read-only clean-commit identity check for CUDA/AMP qualification execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "0d3ac30f2f01042b34b3ece370ffdb6f8d303e99"
IMPLEMENTATION_SHA256 = "8cbf761948770acf7d26cde639b803884a3dc0cf0ed2d878723c0e7368846494"
TEST_SHA256 = "80a3958dca60d6809fd0961bdb67e9b66c987fc2d636e0042069f806a4834654"


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report() -> dict[str, object]:
    """Validate only Git and reviewed-file identities; never invoke CUDA."""
    if _git("status", "--porcelain=v1"):
        raise ValueError("CUDA execution identity requires a clean worktree")
    _git("merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, "HEAD")
    identities = {
        "implementation_sha256": _sha256(
            REPOSITORY_ROOT / "scripts/qualify_vasu_140m_cuda_amp.py"
        ),
        "test_sha256": _sha256(
            REPOSITORY_ROOT / "tests/test_vasu_140m_cuda_amp_qualification.py"
        ),
    }
    if identities["implementation_sha256"] != IMPLEMENTATION_SHA256:
        raise ValueError("reviewed CUDA qualification implementation changed")
    if identities["test_sha256"] != TEST_SHA256:
        raise ValueError("reviewed CUDA qualification tests changed")
    report = {
        "schema": "vasu.model-family-cuda-amp-execution-identity.v1",
        "repository_commit": _git("rev-parse", "HEAD"),
        "implementation_commit": IMPLEMENTATION_COMMIT,
        **identities,
        "worktree_clean": True,
        "cuda_invoked": False,
        "checkpoint_created": False,
        "training_authorized": False,
    }
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
