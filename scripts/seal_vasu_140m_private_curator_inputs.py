"""One-shot CLI for creating encrypted held-out candidate inventories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_private_curator_sealing import (  # noqa: E402
    seal_private_curator_inputs,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-directory",
        type=Path,
        default=Path(r"D:\VASU_PRIVATE_HELDOUT"),
    )
    parser.add_argument(
        "--intake-receipt",
        default="configs/evaluation/vasu_140m_private_curator_intake_20260803.json",
    )
    parser.add_argument(
        "--output-directory",
        default="evaluation/candidates/vasu_140m_base_v2_heldout_curator_v1",
    )
    arguments = parser.parse_args()
    if _git("status", "--porcelain=v1"):
        raise ValueError("held-out sealing requires a clean worktree")
    report = seal_private_curator_inputs(
        repository_root=ROOT,
        private_directory=arguments.private_directory,
        intake_receipt_path=arguments.intake_receipt,
        output_directory=arguments.output_directory,
        repository_commit=_git("rev-parse", "HEAD"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
