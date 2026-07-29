"""Compare two compatible VASU runtime benchmark contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.utils.runtime_benchmark import (  # noqa: E402
    compare_runtime_benchmarks,
    render_runtime_comparison,
    write_runtime_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_runtime_benchmarks(
        baseline=args.baseline, candidate=args.candidate
    )
    if args.output is not None:
        write_runtime_comparison(output=args.output, comparison=report)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_runtime_comparison(report))


if __name__ == "__main__":
    main()
