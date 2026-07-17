"""Check prepared Wikimedia JSONL chunks against a completed FineWeb index."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.deduplication import FineWebDocumentIndex, match_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--wikimedia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.index if args.index.is_absolute() else REPOSITORY_ROOT / args.index
    wikimedia_path = args.wikimedia if args.wikimedia.is_absolute() else REPOSITORY_ROOT / args.wikimedia
    output_path = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    index = FineWebDocumentIndex(index_path)
    matches = []
    try:
        with wikimedia_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                text = record.get("cleaned_text")
                chunk_id = str(record.get("chunk_id", record.get("document_id", line_number)))
                if not isinstance(text, str):
                    raise ValueError(f"Wikimedia record line {line_number} lacks cleaned_text")
                result = match_text(index, text, chunk_id=chunk_id)
                if result.match_type:
                    matches.append(asdict(result))
    finally:
        index.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(matches, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"Compared chunks; overlap candidates: {len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
