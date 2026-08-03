"""Build the VASU-140M FineWeb semantic-contamination quarantine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vasu.data.vasu_140m_semantic_quarantine import (  # noqa: E402
    build_quarantine,
    write_quarantine,
)

DEFAULT_REVIEW = Path(
    "evaluation/results/"
    "vasu_140m_base_v2_semantic_contamination_independent_review_20260803.json"
)
DEFAULT_OUTPUT = Path(
    "configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_quarantine(args.review)
    write_quarantine(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
