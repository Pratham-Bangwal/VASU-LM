"""Create a new versioned runtime benchmark from existing raw profiler output."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vasu.utils.runtime_benchmark import (  # noqa: E402
    create_runtime_benchmark,
    write_runtime_benchmark,
)


def _pairs(values: list[str], *, option: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use NAME=VALUE: {value}")
        name, item = value.split("=", 1)
        if not name or not item or name in result:
            raise ValueError(f"{option} names must be unique and non-empty: {value}")
        result[name] = item
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--metric", action="append", required=True, metavar="NAME=DOT.PATH")
    parser.add_argument("--workload", action="append", required=True, metavar="NAME=VALUE")
    parser.add_argument("--environment", action="append", required=True, metavar="NAME=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = create_runtime_benchmark(
        label=args.label,
        benchmark_kind=args.kind,
        raw_report=args.raw_report,
        metrics=_pairs(args.metric, option="--metric"),
        workload_identity=_pairs(args.workload, option="--workload"),
        environment_identity=_pairs(args.environment, option="--environment"),
    )
    write_runtime_benchmark(output=args.output, benchmark=benchmark)


if __name__ == "__main__":
    main()
