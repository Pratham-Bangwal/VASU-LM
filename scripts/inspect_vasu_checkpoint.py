"""Inspect VASU-60M checkpoints for orchestration without using CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from vasu.training.orchestration import (
    find_latest_valid_checkpoint,
    inspect_checkpoint,
    preserve_milestone,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-finite-check", action="store_true")
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--preserve-milestone", type=Path)
    parser.add_argument("--expected-step", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.checkpoint_dir is not None:
            result = find_latest_valid_checkpoint(
                args.checkpoint_dir,
                verify_finite=not args.no_finite_check,
                calculate_hash=not args.no_hash,
            )
        else:
            result = inspect_checkpoint(
                args.checkpoint,
                verify_finite=not args.no_finite_check,
                calculate_hash=not args.no_hash,
            )
            if not result.valid:
                print(json.dumps(result.to_dict()), file=sys.stderr)
                return 2

        payload = result.to_dict()
        if args.preserve_milestone is not None:
            if args.expected_step is None:
                raise ValueError("--expected-step is required for milestone copy")
            payload["milestone"] = preserve_milestone(
                result.path,
                args.preserve_milestone,
                expected_step=args.expected_step,
            )
        print(json.dumps(payload, sort_keys=True))
        return 0
    except Exception as error:
        print(
            json.dumps(
                {"valid": False, "error": f"{type(error).__name__}: {error}"},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
