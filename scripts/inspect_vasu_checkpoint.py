"""Inspect one VASU checkpoint safely on CPU; never resume or modify it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vasu.training.orchestration import inspect_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skip-finite-check", action="store_true")
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()
    result = inspect_checkpoint(
        args.checkpoint,
        verify_finite=not args.skip_finite_check,
        calculate_hash=not args.skip_hash,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
