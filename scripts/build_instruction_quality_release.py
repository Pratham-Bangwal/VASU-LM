"""Build an atomic, human-approved instruction release and packed masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vasu.data.instruction_quality import build_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = build_release(config)
    print(f"Approved examples: {len(manifest['approved_example_ids'])}")
    print(f"Records: {manifest['records']}")
    print(f"Tokens: {manifest['total_tokens']}")
    print(f"Manifest: {config['release_manifest_path']}")
    print("Training authorized: False")


if __name__ == "__main__":
    main()
