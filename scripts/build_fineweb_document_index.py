"""Inventory, build, resume, or validate the FineWeb document index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.deduplication.fineweb_index import (  # noqa: E402
    FineWebDocumentIndex,
    build_fineweb_document_index,
    load_index_config,
    sha256_file,
)


DEFAULT_CONFIG = Path("configs/data/deduplication/fineweb_index.json")


def classify_artifacts(root: Path) -> list[dict[str, object]]:
    artifacts = [
        ("data/raw/pretrain/fineweb_1m.jsonl", "document-level recoverable; incomplete provenance; reusable"),
        ("data/processed/pretrain/fineweb_1m.bin", "token-only and unsuitable"),
        ("data/processed/pretrain/fineweb_extension_500m.bin", "token-only and unsuitable"),
        ("data/processed/pretrain/fineweb_extension_500m.bin.dedup.sqlite3", "reusable exact hashes; incomplete for near matching"),
        ("data/processed/pretrain/fineweb_extension_500m_metadata.json", "reusable provenance metadata"),
        ("data/processed/pretrain/fineweb_manifest.json", "reusable token-layout metadata; no document boundaries"),
    ]
    return [
        {
            "path": path,
            "exists": (root / path).is_file(),
            "classification": classification if (root / path).is_file() else "missing",
        }
        for path, classification in artifacts
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPOSITORY_ROOT / args.config
    config = load_index_config(config_path)
    if args.inventory:
        print(json.dumps(classify_artifacts(REPOSITORY_ROOT), indent=2))
        return 0
    if args.dry_run:
        print("FineWeb document index dry-run: configuration valid")
        print(f"Configured sources: {len(config.sources)}")
        print(f"Bounded maximum documents: {config.maximum_documents}")
        print("Full index build performed: no")
        return 0
    output = REPOSITORY_ROOT / config.output_path
    if args.validate:
        metadata = json.loads((REPOSITORY_ROOT / config.metadata_path).read_text(encoding="utf-8"))
        if sha256_file(output) != metadata["output_sha256"]:
            raise ValueError("FineWeb document index output hash mismatch")
        index = FineWebDocumentIndex(output)
        index.close()
        print(f"FineWeb document index valid: {output}")
        return 0
    metadata = build_fineweb_document_index(
        config, repository_root=REPOSITORY_ROOT, resume=args.resume
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
