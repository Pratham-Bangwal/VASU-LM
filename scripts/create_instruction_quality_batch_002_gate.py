"""Create the deterministic Batch 002 human-review gate sample."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE = Path(
    "data/raw/instruct/"
    "vasu_instruction_quality_v1_batch_002.jsonl"
)
OUTPUT_DIR = Path(
    "evaluation/results/"
    "instruction_quality_batch_002_gate_v2"
)

SAMPLE_PATH = OUTPUT_DIR / (
    "vasu_instruction_quality_v1_batch_002_gate_v2_sample.jsonl"
)
MANIFEST_PATH = OUTPUT_DIR / (
    "vasu_instruction_quality_v1_batch_002_gate_v2_manifest.json"
)
PACKET_PATH = OUTPUT_DIR / (
    "vasu_instruction_quality_v1_batch_002_gate_v2_review_packet.txt"
)

SEED = 42
TARGET_COUNTS = {
    "short_factual_qa": 25,
    "beginner_explanation": 20,
    "exact_format_following": 20,
    "rewriting_transformation": 15,
    "lists_structured_output": 10,
    "json_schema_output": 5,
    "uncertainty_honest_fallback": 5,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def selection_key(record: dict[str, Any]) -> tuple[str, str]:
    payload = f"{SEED}:{record['example_id']}".encode("utf-8")
    return (
        hashlib.sha256(payload).hexdigest(),
        record["example_id"],
    )


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            file.write("\n")


def create_review_packet(
    records: list[dict[str, Any]],
    source_sha256: str,
) -> str:
    lines = [
        "VASU INSTRUCTION QUALITY V1",
        "BATCH 002 - GATE V2 HUMAN REVIEW",
        "",
        f"Source SHA-256: {source_sha256}",
        f"Sample records: {len(records)}",
        "",
        "Review every record for:",
        "- semantic correctness",
        "- instruction adherence",
        "- response usefulness",
        "- natural wording",
        "- unwanted repetition",
        "- format correctness",
        "- safe uncertainty handling",
        "",
        "Do not approve training from this packet alone.",
    ]

    for record in records:
        lines.extend(
            [
                "",
                "=" * 88,
                f"Example ID: {record['example_id']}",
                f"Capability: {record['capability']}",
                f"Difficulty: {record['difficulty']}",
                f"Instruction: {record['instruction']}",
                f"Input: {record['input']}",
                f"Response: {record['response']}",
                "Format constraints: "
                + json.dumps(
                    record["format_constraints"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "Facts: "
                + json.dumps(
                    record["facts"],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                f"Source reference: {record['source_reference']}",
                f"Record SHA-256: {record_sha256(record)}",
                "Human-review decision: ",
                "Human-review notes: ",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"source not found: {SOURCE}")

    records = load_jsonl(SOURCE)

    if len(records) != 500:
        raise ValueError(
            f"expected 500 source records, found {len(records)}"
        )

    source_counts = Counter(
        record["capability"]
        for record in records
    )

    selected: list[dict[str, Any]] = []

    for capability, target_count in TARGET_COUNTS.items():
        candidates = [
            record
            for record in records
            if record["capability"] == capability
        ]

        if len(candidates) < target_count:
            raise ValueError(
                f"{capability} has {len(candidates)} records; "
                f"{target_count} required"
            )

        candidates.sort(key=selection_key)
        selected.extend(candidates[:target_count])

    selected.sort(key=lambda record: record["example_id"])

    actual_counts = Counter(
        record["capability"]
        for record in selected
    )

    if len(selected) != 100:
        raise AssertionError(
            f"expected 100 selected records, found {len(selected)}"
        )

    if dict(actual_counts) != TARGET_COUNTS:
        raise AssertionError(
            f"unexpected sample counts: {dict(actual_counts)}"
        )

    selected_ids = [
        record["example_id"]
        for record in selected
    ]

    if len(selected_ids) != len(set(selected_ids)):
        raise AssertionError("duplicate sample IDs detected")

    source_sha256 = sha256_file(SOURCE)

    manifest = {
        "schema_version": (
            "vasu_instruction_quality_gate_manifest_v1"
        ),
        "dataset_name": (
            "vasu_instruction_quality_v1_batch_002"
        ),
        "gate_name": (
            "instruction_quality_batch_002_gate_v2"
        ),
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "selection_method": (
            "stratified_sha256_seeded"
        ),
        "selection_seed": SEED,
        "source_path": SOURCE.as_posix(),
        "source_sha256": source_sha256,
        "source_record_count": len(records),
        "source_category_counts": dict(source_counts),
        "sample_path": SAMPLE_PATH.as_posix(),
        "sample_sha256": None,
        "sample_size": len(selected),
        "target_counts": TARGET_COUNTS,
        "actual_counts": dict(actual_counts),
        "selected_ids": selected_ids,
        "selected_record_sha256": {
            record["example_id"]: record_sha256(record)
            for record in selected
        },
        "review_decisions_transferred": 0,
        "automatic_approval": False,
        "training_authorized": False,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(SAMPLE_PATH, selected)

    manifest["sample_sha256"] = sha256_file(SAMPLE_PATH)

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    PACKET_PATH.write_text(
        create_review_packet(
            selected,
            source_sha256,
        ),
        encoding="utf-8",
        newline="\n",
    )

    print(f"Source records: {len(records)}")
    print(f"Gate sample records: {len(selected)}")
    print(f"Gate counts: {dict(actual_counts)}")
    print(f"Source SHA-256: {source_sha256}")
    print(f"Sample SHA-256: {manifest['sample_sha256']}")
    print(f"Sample: {SAMPLE_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Review packet: {PACKET_PATH}")
    print("Review decisions transferred: 0")
    print("Training authorized: False")


if __name__ == "__main__":
    main()
