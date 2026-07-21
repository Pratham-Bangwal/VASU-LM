"""Inspect and record hash-bound human decisions for instruction examples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from vasu.data.instruction_quality import (
    REVIEW_STATUSES,
    load_jsonl,
    load_review_decisions,
    make_review_decision,
    transition_allowed,
    validate_record,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", choices=REVIEW_STATUSES)
    parser.add_argument("--capability")
    parser.add_argument("--finding")
    parser.add_argument("--example-id")
    parser.add_argument("--set-status", choices=REVIEW_STATUSES)
    parser.add_argument("--notes")
    parser.add_argument("--reviewer", default="human")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training_authorized") is not False:
        raise ValueError("training_authorized must remain false")
    records = load_jsonl(Path(config["source_path"]))
    by_id = {record["example_id"]: record for record in records}
    review_path = Path(config["review_path"])
    decisions = load_review_decisions(review_path)

    if args.set_status:
        if not args.example_id or args.example_id not in by_id:
            raise ValueError("--set-status requires a valid --example-id")
        current = decisions.get(args.example_id, {}).get("status", "unreviewed")
        if not transition_allowed(current, args.set_status):
            raise ValueError(f"status transition not allowed: {current} -> {args.set_status}")
        decisions[args.example_id] = make_review_decision(
            by_id[args.example_id], args.set_status, args.reviewer, args.notes
        )
        write_jsonl(review_path, [decisions[key] for key in sorted(decisions)])
        print(f"Updated {args.example_id}: {args.set_status}")

    selected = records
    if args.example_id:
        selected = [by_id[args.example_id]] if args.example_id in by_id else []
    if args.status:
        selected = [record for record in selected if decisions.get(record["example_id"], {}).get("status", "unreviewed") == args.status]
    if args.capability:
        selected = [record for record in selected if record["capability"] == args.capability]
    if args.finding:
        selected = [record for record in selected if any(finding.code == args.finding for finding in validate_record(record))]
    if args.list or args.example_id:
        for record in selected:
            status = decisions.get(record["example_id"], {}).get("status", "unreviewed")
            print(f"{record['example_id']} [{status}] {record['capability']}: {record['instruction']}")
            if args.example_id:
                print(json.dumps(record, indent=2, ensure_ascii=False))
                print("Decision:", json.dumps(decisions.get(record["example_id"]), indent=2))
    if args.progress:
        statuses = Counter(decisions.get(record["example_id"], {}).get("status", "unreviewed") for record in records)
        capabilities = Counter(record["capability"] for record in records)
        print("Review status:", dict(statuses))
        print("Capabilities:", dict(capabilities))
    if args.export:
        lines = ["VASU INSTRUCTION QUALITY REVIEW PACKET", ""]
        for record in selected:
            lines.extend(
                [
                    "=" * 80,
                    f"ID: {record['example_id']}",
                    f"Capability: {record['capability']}",
                    f"Instruction: {record['instruction']}",
                    f"Input: {record['input']}",
                    f"Response: {record['response']}",
                    f"Findings: {[finding.code for finding in validate_record(record)]}",
                    "Decision: ",
                    "Notes: ",
                ]
            )
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Exported: {args.export}")


if __name__ == "__main__":
    main()
