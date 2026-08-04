"""Non-executing identity smoke for development-inventory promotion."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "7a47562a80d47f939cdd3df7344058d0e2973856"
BINDINGS = {
    "implementation": (
        Path("evaluation/framework/vasu_140m_development_promotion.py"),
        "34e2d245858cd422f4a272de0e07ca908f26284b4ed3ff804fa62d6bd7c12ddd",
    ),
    "tests": (
        Path("tests/test_vasu_140m_development_promotion.py"),
        "4e486737a242a285c87e9380d2d91ad4335f065df4f15ea97fb5ead601a3c8a4",
    ),
    "acceptance": (
        Path(
            "docs/VASU_140M_DEVELOPMENT_INVENTORY_PROMOTION_IMPLEMENTATION_"
            "INDEPENDENT_REVIEW_DECISION_20260804.md"
        ),
        "3559e5dfda4283827015fea60ca45e9a575a740d14a6886766cfa6f95ccf13ac",
    ),
}
OUTPUT = Path("evaluation/candidates/vasu_140m_base_v2_development_v1")


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def build_report() -> dict[str, object]:
    if _git("status", "--porcelain=v1"):
        raise ValueError("development promotion execution requires a clean worktree")
    head = _git("rev-parse", "HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, head],
        cwd=ROOT,
        check=True,
    )
    identities = {}
    for label, (path, expected) in BINDINGS.items():
        observed = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"{label} identity mismatch")
        identities[label] = observed
    if (ROOT / OUTPUT).exists():
        raise FileExistsError("development promotion output already exists")
    return {
        "schema_id": "vasu_140m_development_promotion_execution_identity_v1",
        "runtime_commit": head,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "identities": identities,
        "bound_command": "python scripts\\promote_vasu_140m_development_inventories.py",
        "bound_output": OUTPUT.as_posix(),
        "promotion_invoked": False,
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))
