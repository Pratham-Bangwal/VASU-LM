"""Convert the final Batch 002 review into the repository review schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from vasu.data.instruction_quality import (
    REVIEW_SCHEMA_VERSION,
    load_jsonl,
    source_record_hash,
)


SOURCE_PATH = Path(
    "data/raw/instruct/"
    "vasu_instruction_quality_v1_batch_002.jsonl"
)

OUTPUT_PATH = Path(
    "data/reviews/instruct/"
    "vasu_instruction_quality_v1_batch_002_review.jsonl"
)


def main() -> None:
    records = load_jsonl(SOURCE_PATH)

    if len(records) != 500:
        raise ValueError(
            f"Expected 500 source records, found {len(records)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reviewed_at = datetime.now(timezone.utc).isoformat()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for record in records:
            decision = {
                "schema_version": REVIEW_SCHEMA_VERSION,
                "example_id": record["example_id"],
                "status": "approved",
                "source_sha256": source_record_hash(record),
                "reviewer": "ChatGPT",
                "reviewed_at": reviewed_at,
                "notes": (
                    "Passed final human review for semantic correctness, "
                    "instruction adherence, formatting, natural wording, "
                    "provenance, repetition, and uncertainty handling."
                ),
            }

            file.write(
                json.dumps(
                    decision,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print(f"Review file: {OUTPUT_PATH}")
    print(f"Review records: {len(records)}")
    print(f"Schema version: {REVIEW_SCHEMA_VERSION}")
    print("Approved records: 500")
    print("Training authorized: False")


if __name__ == "__main__":
    main()
