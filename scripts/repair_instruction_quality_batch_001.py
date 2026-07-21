"""Rebuild batch 001 after gate-v1 feedback and create a fresh gate sample."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from author_instruction_quality_batch_001 import build_artifacts
from vasu.data.instruction_quality import (
    atomic_write_text,
    load_jsonl,
    source_record_hash,
    validate_record,
    write_jsonl,
)
from vasu.data.instruction_quality_batch_001 import EXPECTED_COUNTS, author_batch


GATE_COUNTS = {
    "short_factual_qa": 25,
    "beginner_explanation": 20,
    "exact_format_following": 20,
    "rewriting_transformation": 15,
    "lists_structured_output": 10,
    "json_schema_output": 5,
    "uncertainty_honest_fallback": 5,
}
SEED = 42


def _stable_key(example_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{example_id}".encode()).hexdigest()


def select_gate_sample(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the exact stratified gate sample deterministically."""
    selected = []
    for capability, count in GATE_COUNTS.items():
        candidates = sorted(
            (row for row in records if row["capability"] == capability),
            key=lambda row: (_stable_key(row["example_id"]), row["example_id"]),
        )
        if len(candidates) < count:
            raise ValueError(f"not enough {capability} records for gate sample")
        selected.extend(candidates[:count])
    return sorted(selected, key=lambda row: row["example_id"])


def _packet(
    sample: list[dict[str, Any]],
    old_decisions: dict[str, dict[str, Any]],
    new_hashes: dict[str, str],
    changed_ids: set[str],
) -> str:
    lines = [
        "VASU INSTRUCTION QUALITY BATCH 001 - GATE V2 REVIEW PACKET",
        "No gate-v1 decision is transferred. Human decision fields are blank.",
    ]
    for row in sample:
        example_id = row["example_id"]
        old_gate_hash = old_decisions.get(example_id, {}).get("source_sha256")
        lines.extend(
            [
                "",
                "=" * 88,
                f"Example ID: {example_id}",
                f"Capability: {row['capability']}",
                f"Instruction: {row['instruction']}",
                f"Input: {row['input']}",
                f"Response: {row['response']}",
                "Format constraints: "
                + json.dumps(row["format_constraints"], ensure_ascii=False, sort_keys=True),
                "Factual metadata: "
                + json.dumps(row["facts"], ensure_ascii=False, sort_keys=True),
                f"Exact source URL: {row['source_reference']}",
                f"Old gate-v1 source hash: {old_gate_hash}",
                f"New source hash: {new_hashes[example_id]}",
                f"Record changed: {example_id in changed_ids}",
                "Automatic findings: "
                + json.dumps(
                    [finding.__dict__ for finding in validate_record(row)],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "Human-review decision: ",
                "Human-review notes: ",
            ]
        )
    return "\n".join(lines) + "\n"


def repair_batch(config_path: Path, gate_v1_dir: Path, gate_v2_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("training_authorized") is not False:
        raise ValueError("training_authorized must remain false")
    source_path = Path(config["source_path"])
    review_path = Path(config["review_path"])
    if review_path.exists() and review_path.stat().st_size:
        raise ValueError("production review file must remain empty")
    old_records = load_jsonl(source_path)
    new_records = author_batch()
    old_ids = [row["example_id"] for row in old_records]
    new_ids = [row["example_id"] for row in new_records]
    if old_ids != new_ids:
        raise ValueError("repair changed or reordered stable example IDs")
    new_hashes = {row["example_id"]: source_record_hash(row) for row in new_records}
    repair_audit_path = Path(
        "data/manifests/instruct/vasu_instruction_quality_v1_batch_001_repair_audit.json"
    )
    already_repaired = old_records == new_records and repair_audit_path.exists()
    if already_repaired:
        prior_audit = json.loads(repair_audit_path.read_text(encoding="utf-8"))
        old_source_sha = prior_audit["old_source_sha256"]
        changed_ids = list(prior_audit["changed_example_ids"])
        unchanged_ids = list(prior_audit["unchanged_example_ids"])
    else:
        old_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        old_hashes = {
            row["example_id"]: source_record_hash(row) for row in old_records
        }
        changed_ids = [
            example_id
            for example_id in old_ids
            if old_hashes[example_id] != new_hashes[example_id]
        ]
        unchanged_ids = [
            example_id
            for example_id in old_ids
            if old_hashes[example_id] == new_hashes[example_id]
        ]

    gate_review_path = gate_v1_dir / "vasu_instruction_quality_v1_batch_001_gate_review.jsonl"
    old_decision_rows = load_jsonl(gate_review_path)
    old_decisions = {row["example_id"]: row for row in old_decision_rows}
    # Gate v1 used a different JSON serialization for record hashes. Content
    # comparison, not cross-convention hash equality, determines staleness.
    changed_id_set = set(changed_ids)
    stale_decisions = [
        example_id for example_id in old_decisions if example_id in changed_id_set
    ]

    if not already_repaired:
        write_jsonl(source_path, new_records)
    report = build_artifacts(config_path, regenerate_source=False)
    new_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if report["source_sha256"] != new_source_sha:
        raise AssertionError("post-repair report source hash mismatch")

    gate_v2_dir.mkdir(parents=True, exist_ok=True)
    sample = select_gate_sample(new_records)
    sample_path = gate_v2_dir / "vasu_instruction_quality_v1_batch_001_gate_v2_sample.jsonl"
    manifest_path = gate_v2_dir / "vasu_instruction_quality_v1_batch_001_gate_v2_manifest.json"
    packet_path = gate_v2_dir / "vasu_instruction_quality_v1_batch_001_gate_v2_review_packet.txt"
    write_jsonl(sample_path, sample)
    manifest = {
        "schema_version": "vasu_instruction_quality_gate_sample_v2",
        "source_file": str(source_path),
        "old_source_sha256": old_source_sha,
        "source_sha256": new_source_sha,
        "selection_method": "Per-capability SHA-256 ordering of '42:<example_id>'",
        "seed": SEED,
        "sample_size": len(sample),
        "target_counts": GATE_COUNTS,
        "actual_counts": dict(Counter(row["capability"] for row in sample)),
        "selected_ids": [row["example_id"] for row in sample],
        "selected_record_hashes": {
            row["example_id"]: new_hashes[row["example_id"]] for row in sample
        },
        "changed_example_ids": changed_ids,
        "changed_records": len(changed_ids),
        "unchanged_records": len(unchanged_ids),
        "gate_v1_decisions": len(old_decisions),
        "stale_gate_v1_decisions": len(stale_decisions),
        "stale_gate_v1_example_ids": sorted(stale_decisions),
        "review_decisions_transferred": 0,
        "training_authorized": False,
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    atomic_write_text(
        packet_path,
        _packet(sample, old_decisions, new_hashes, changed_id_set),
    )
    if not already_repaired:
        source_changes = []
        old_by_id = {row["example_id"]: row for row in old_records}
        for row in new_records[: EXPECTED_COUNTS["short_factual_qa"]]:
            old_row = old_by_id[row["example_id"]]
            source_changes.append(
                {
                    "example_id": row["example_id"],
                    "old_source_reference": old_row["source_reference"],
                    "new_source_reference": row["source_reference"],
                    "source_replaced": old_row["source_reference"]
                    != row["source_reference"],
                }
            )
        repair_audit = {
            "old_source_sha256": old_source_sha,
            "new_source_sha256": new_source_sha,
            "records_changed": len(changed_ids),
            "records_unchanged": len(unchanged_ids),
            "changed_example_ids": changed_ids,
            "unchanged_example_ids": unchanged_ids,
            "gate_v1_decisions_stale": len(stale_decisions),
            "stale_gate_v1_example_ids": sorted(stale_decisions),
            "factual_records_audited": 125,
            "factual_source_changes": source_changes,
            "production_review_entries": 0,
            "training_authorized": False,
        }
        atomic_write_text(
            repair_audit_path,
            json.dumps(repair_audit, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
        )
    else:
        prior_audit["gate_v1_decisions_stale"] = len(stale_decisions)
        prior_audit["stale_gate_v1_example_ids"] = sorted(stale_decisions)
        atomic_write_text(
            repair_audit_path,
            json.dumps(prior_audit, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
        )
    return {
        **manifest,
        "sample_path": str(sample_path),
        "manifest_path": str(manifest_path),
        "packet_path": str(packet_path),
        "repair_audit_path": str(repair_audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-v1-dir", type=Path, required=True)
    parser.add_argument("--gate-v2-dir", type=Path, required=True)
    args = parser.parse_args()
    result = repair_batch(args.config, args.gate_v1_dir, args.gate_v2_dir)
    print(f"Old source SHA-256: {result['old_source_sha256']}")
    print(f"New source SHA-256: {result['source_sha256']}")
    print(f"Records changed: {result['changed_records']}")
    print(f"Records unchanged: {result['unchanged_records']}")
    print(f"Gate-v1 decisions stale: {result['stale_gate_v1_decisions']}")
    print(f"Gate-v2 sample: {result['sample_path']}")
    print("Review decisions transferred: 0")
    print("Training authorized: False")


if __name__ == "__main__":
    main()
