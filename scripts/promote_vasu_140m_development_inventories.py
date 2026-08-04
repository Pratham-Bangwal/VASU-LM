"""CLI for the reviewed VASU-140M development promotion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_development_promotion import (  # noqa: E402
    promote_development_inventories,
)


def main() -> None:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    report = promote_development_inventories(ROOT, commit)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
