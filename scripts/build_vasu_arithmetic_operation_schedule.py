"""Build a train-only operation-aware arithmetic-v2 view and small schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vasu.data.arithmetic_operation_schedule import (
    build_operation_schedule,
    build_operation_view,
    validate_operation_schedule,
    validate_operation_view,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-dir", type=Path, required=True)
    parser.add_argument("--schedule-dir", type=Path, required=True)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = {
            "view": validate_operation_view(args.view_dir),
            "schedule": validate_operation_schedule(args.schedule_dir),
        }
    else:
        view = build_operation_view(
            train_records_path=Path(
                "data/processed/capability/verified_arithmetic_v2/train_records.jsonl"
            ),
            tokenizer_path=Path("assets/tokenizer.json"),
            output_dir=args.view_dir,
            seed=args.seed,
        )
        weights = {
            operation: 1 / 11
            for operation in (
                "addition",
                "subtraction",
                "multiplication",
                "exact_division",
                "percentage",
                "fraction",
                "mixed_expression",
                "sequence",
                "word_problem",
                "comparison",
                "numeric_property",
            )
        }
        schedule = build_operation_schedule(
            view_dir=args.view_dir,
            output_dir=args.schedule_dir,
            total_records=args.records,
            operation_weights=weights,
            seed=args.seed,
            with_replacement=True,
        )
        result = {"view": view, "schedule": schedule}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
