"""Print an identity-bound paired comparison of two completed arithmetic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.comparison import compare_verified_arithmetic_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_verified_arithmetic_runs(
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
