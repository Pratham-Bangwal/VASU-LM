"""Validate private held-out inputs and print only hash/count evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_private_curator_intake import (  # noqa: E402
    validate_private_curator_intake,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-directory",
        type=Path,
        default=Path(r"D:\VASU_PRIVATE_HELDOUT"),
    )
    arguments = parser.parse_args()
    report = validate_private_curator_intake(
        ROOT,
        arguments.private_directory,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
