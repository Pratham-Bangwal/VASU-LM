"""Compare two compatible frozen evaluation snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.frozen_snapshots import (  # noqa: E402
    compare_frozen_evaluation_snapshots,
    render_frozen_snapshot_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "text"), default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_frozen_evaluation_snapshots(
        baseline=args.baseline, candidate=args.candidate
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_frozen_snapshot_comparison(report))


if __name__ == "__main__":
    main()
