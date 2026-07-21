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
    parser.add_argument(
        "--additional-index",
        type=Path,
        default=None,
        help="Optional second compatible index; emits separate and combined statistics.",
    )
    parser.add_argument("--wikimedia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.index if args.index.is_absolute() else REPOSITORY_ROOT / args.index
    wikimedia_path = args.wikimedia if args.wikimedia.is_absolute() else REPOSITORY_ROOT / args.wikimedia
    output_path = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    additional_path = None
    if args.additional_index is not None:
        additional_path = (
            args.additional_index
            if args.additional_index.is_absolute()
            else REPOSITORY_ROOT / args.additional_index
        )
    indexes = [("original", FineWebDocumentIndex(index_path))]
    if additional_path is not None:
        indexes.append(("extension", FineWebDocumentIndex(additional_path)))
    matches: list[dict[str, object]] = []
    checked = 0
    try:
        with wikimedia_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                text = record.get("cleaned_text")
                chunk_id = str(record.get("chunk_id", record.get("document_id", line_number)))
                if not isinstance(text, str):
                    raise ValueError(f"Wikimedia record line {line_number} lacks cleaned_text")
                checked += 1
                for label, index in indexes:
                    result = match_text(index, text, chunk_id=chunk_id)
                    if result.match_type:
                        matches.append({"index": label, **asdict(result)})
    finally:
        for _, index in indexes:
            index.close()
    if additional_path is None:
        output_value: object = [
            {key: value for key, value in item.items() if key != "index"}
            for item in matches
        ]
    else:
        def counts(label: str) -> dict[str, int]:
            selected = [item for item in matches if item["index"] == label]
            return {
                "exact_matches": sum(item["match_type"] == "exact" for item in selected),
                "high_confidence_near_matches": sum(item["match_type"] == "near" for item in selected),
                "ambiguous_candidates": sum(item["match_type"] == "ambiguous" for item in selected),
            }

        original = counts("original")
        extension = counts("extension")
        matched_chunks = {str(item["wikimedia_chunk_id"]) for item in matches}
        output_value = {
            "format_version": 1,
            "chunks_checked": checked,
            "original": original,
            "extension": extension,
            "combined": {
                "exact_matches": original["exact_matches"] + extension["exact_matches"],
                "high_confidence_near_matches": (
                    original["high_confidence_near_matches"]
                    + extension["high_confidence_near_matches"]
                ),
                "ambiguous_candidates": (
                    original["ambiguous_candidates"] + extension["ambiguous_candidates"]
                ),
                "no_match_chunks": checked - len(matched_chunks),
            },
            "matches": matches,
            "review_only": True,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(output_value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"Compared chunks; overlap candidates: {len(matches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
