"""Validate and report a VASU instruction-quality source collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vasu.data.instruction_quality import (
    atomic_write_text,
    load_jsonl,
    load_review_decisions,
    sha256_file,
    validation_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training_authorized") is not False:
        raise ValueError("training_authorized must remain false")
    source = Path(config["source_path"])
    review = Path(config["review_path"])
    records = load_jsonl(source)
    decisions = load_review_decisions(review)
    report = validation_report(
        records,
        decisions,
        config["target_distribution"],
        float(config["near_duplicate_threshold"]),
    )
    report.update(
        {
            "dataset_name": config["dataset_name"],
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "review_path": str(review),
            "review_sha256": sha256_file(review) if review.exists() else None,
            "training_authorized": False,
        }
    )
    report_json = Path(config["validation_report_json"])
    report_text = Path(config["validation_report_text"])
    atomic_write_text(
        report_json,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    lines = [
        "VASU INSTRUCTION QUALITY VALIDATION",
        f"Total records: {report['total_records']}",
        f"Valid records: {report['valid_records']}",
        f"Invalid records: {report['invalid_records']}",
        f"Review statuses: {report['review_status_counts']}",
        f"Category counts: {report['category_counts']}",
        f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
        f"Near-duplicate candidates: {len(report['near_duplicate_candidates'])}",
        f"Automatic approval: {report['automatic_approval']}",
    ]
    atomic_write_text(report_text, "\n".join(lines) + "\n")
    source_manifest = {
        "schema_version": config["schema_version"],
        "dataset_name": config["dataset_name"],
        "source_path": str(source),
        "source_sha256": report["source_sha256"],
        "records": len(records),
        "training_authorized": False,
        "status": "demonstration fixture; human approval incomplete",
    }
    atomic_write_text(
        Path(config["source_manifest_path"]),
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
    )
    review_manifest = {
        "schema_version": "vasu_instruction_quality_review_manifest_v1",
        "review_path": str(review),
        "review_sha256": report["review_sha256"],
        "decisions": len(decisions),
        "status_counts": report["review_status_counts"],
        "training_authorized": False,
    }
    atomic_write_text(
        Path(config["review_manifest_path"]),
        json.dumps(review_manifest, indent=2, sort_keys=True) + "\n",
    )
    print("\n".join(lines))
    print(f"JSON report: {report_json}")
    print(f"Text report: {report_text}")


if __name__ == "__main__":
    main()
