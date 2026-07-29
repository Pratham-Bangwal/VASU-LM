"""Create a new hash-bound evaluation snapshot without overwriting artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.frozen_snapshots import (  # noqa: E402
    create_frozen_evaluation_snapshot,
    write_frozen_evaluation_snapshot,
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
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--metric", action="append", required=True, metavar="NAME=SOURCE:DOT.PATH"
    )
    parser.add_argument("--identity", action="append", required=True, metavar="NAME=VALUE")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pairs = _pairs(args.source, option="--source")
    metric_pairs = _pairs(args.metric, option="--metric")
    parsed_metrics: dict[str, tuple[str, str]] = {}
    for name, value in metric_pairs.items():
        if ":" not in value:
            raise ValueError(f"--metric must use SOURCE:DOT.PATH: {name}={value}")
        source, path = value.split(":", 1)
        if not source or not path:
            raise ValueError(f"--metric must use SOURCE:DOT.PATH: {name}={value}")
        parsed_metrics[name] = (source, path)
    snapshot = create_frozen_evaluation_snapshot(
        label=args.label,
        sources={name: Path(path) for name, path in source_pairs.items()},
        metrics=parsed_metrics,
        comparison_identity=_pairs(args.identity, option="--identity"),
    )
    write_frozen_evaluation_snapshot(output=args.output, snapshot=snapshot)


if __name__ == "__main__":
    main()
