"""Validate an existing verified-arithmetic v1 release without regeneration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from scripts.serialize_vasu_verified_arithmetic_v1 import validate_release
else:
    from serialize_vasu_verified_arithmetic_v1 import validate_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--compact-json-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest.name != "manifest.json":
        raise SystemExit("--manifest must reference manifest.json")
    manifest = validate_release(
        args.manifest.parent,
        tokenizer_path=args.tokenizer,
    )
    if args.compact_json_summary:
        output = {
            "dataset_id": manifest["dataset_id"],
            "release_status": manifest["release_status"],
            "valid": True,
            "training_authorized": manifest["training_authorized"],
        }
        print(json.dumps(output, separators=(",", ":")))
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
