"""Build and audit the deterministic 500-record instruction candidate batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vasu.data.instruction_quality import (
    atomic_write_text,
    load_jsonl,
    sha256_file,
)
from vasu.data.instruction_quality_batch_002 import (
    AUTHORING_VERSION,
    EXPECTED_COUNTS,
    author_batch,
    response_word_statistics,
    source_domains,
    token_length_audit,
    validate_authored_batch,
    write_batch,
)


def _review_packet(records: list[dict], report: dict) -> str:
    findings_by_id: dict[str, list[dict]] = {}
    for finding in report["findings"]:
        findings_by_id.setdefault(finding["example_id"], []).append(finding)
    lines = [
        "VASU INSTRUCTION QUALITY V1 - BATCH 002 REVIEW PACKET",
        "Automatic validation does not constitute approval.",
        "All human decision and note fields are intentionally blank.",
    ]
    for capability in EXPECTED_COUNTS:
        lines.extend(["", "#" * 88, f"CAPABILITY: {capability}", "#" * 88])
        for record in records:
            if record["capability"] != capability:
                continue
            score = report["quality_scores"][record["example_id"]]
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
                    + json.dumps(record["format_constraints"], ensure_ascii=False, sort_keys=True),
                    "Facts: " + json.dumps(record["facts"], ensure_ascii=False, sort_keys=True),
                    f"Source reference: {record['source_reference']}",
                    "Automatic findings: "
                    + json.dumps(findings_by_id.get(record["example_id"], []), ensure_ascii=False),
                    "Quality score: " + json.dumps(score, ensure_ascii=False, sort_keys=True),
                    "Human-review decision: ",
                    "Human-review notes: ",
                ]
            )
    return "\n".join(lines) + "\n"


def build_artifacts(config_path: Path, *, regenerate_source: bool) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("training_authorized") is not False:
        raise ValueError("training_authorized must remain false")
    source_path = Path(config["source_path"])
    demo_path = Path(config["demo_path"])
    review_path = Path(config["review_path"])
    tokenizer_path = Path(config["tokenizer_path"])
    if sha256_file(demo_path) != config["demo_sha256"]:
        raise ValueError("demonstration fixture SHA-256 mismatch")
    if sha256_file(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("tokenizer SHA-256 mismatch")
    if regenerate_source and review_path.exists() and review_path.read_bytes():
        raise ValueError(
            "cannot regenerate the source after human review decisions exist; "
            "use --artifacts-only instead"
        )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.touch(exist_ok=True)
    if regenerate_source:
        records = write_batch(source_path)
    else:
        records = load_jsonl(source_path)
        if records != author_batch():
            raise ValueError("source JSONL differs from deterministic authoring output")
    demo = load_jsonl(demo_path)
    report = validate_authored_batch(
        records,
        demo,
        float(config["near_duplicate_threshold"]),
    )
    token_audit = token_length_audit(
        records,
        tokenizer_path,
        int(config["sequence_length"]),
    )
    report.update(
        {
            "dataset_name": config["dataset_name"],
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "review_path": str(review_path),
            "review_sha256": sha256_file(review_path),
            "configuration_path": str(config_path),
            "configuration_sha256": sha256_file(config_path),
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "token_length_audit": token_audit,
            "response_word_counts": response_word_statistics(records),
            "source_reference_domains": source_domains(records),
            "training_authorized": False,
            "automatic_approval": False,
        }
    )
    blockers = [
        bool(report["invalid_records"]),
        bool(report["structural_errors"]),
        bool(report["exact_duplicate_groups"]),
        bool(report["near_duplicate_candidates"]),
        bool(report["duplicates_against_demo"]["exact_duplicate_groups"]),
        bool(report["duplicates_against_demo"]["near_duplicate_candidates"]),
        bool(token_audit["truncated_example_ids"]),
    ]
    if any(blockers):
        raise ValueError("authored batch failed one or more production-candidate gates")
    report_json_path = Path(config["validation_report_json"])
    report_text_path = Path(config["validation_report_text"])
    atomic_write_text(
        report_json_path,
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    text_lines = [
        "VASU INSTRUCTION QUALITY V1 - BATCH 002 VALIDATION",
        f"Total records: {report['total_records']}",
        f"Valid records: {report['valid_records']}",
        f"Invalid records: {report['invalid_records']}",
        f"Review statuses: {report['review_status_counts']}",
        f"Category counts: {report['category_counts']}",
        f"Difficulty counts: {report['difficulty_counts']}",
        f"Language counts: {report['language_counts']}",
        f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
        f"Near-duplicate candidates: {len(report['near_duplicate_candidates'])}",
        "Duplicates against demo: "
        f"exact={len(report['duplicates_against_demo']['exact_duplicate_groups'])}, "
        f"near={len(report['duplicates_against_demo']['near_duplicate_candidates'])}",
        f"Maximum full-example tokens: {token_audit['full_example_tokens']['maximum']}",
        f"Truncated examples: {len(token_audit['truncated_example_ids'])}",
        "Automatic approval: False",
        "Training authorized: False",
    ]
    atomic_write_text(report_text_path, "\n".join(text_lines) + "\n")
    source_manifest = {
        "schema_version": config["schema_version"],
        "dataset_name": config["dataset_name"],
        "source_path": str(source_path),
        "source_sha256": report["source_sha256"],
        "record_count": len(records),
        "category_counts": report["category_counts"],
        "difficulty_counts": report["difficulty_counts"],
        "language_counts": report["language_counts"],
        "creation_method": config["creation_method"],
        "creation_timestamp": config["creation_timestamp"],
        "authoring_version": AUTHORING_VERSION,
        "tokenizer_audit_hash": report["tokenizer_sha256"],
        "configuration_sha256": report["configuration_sha256"],
        "template_family_counts": report["template_family_counts"],
        "training_authorized": False,
        "status": "unreviewed production candidates; no tokenized release",
    }
    atomic_write_text(
        Path(config["source_manifest_path"]),
        json.dumps(source_manifest, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
    )
    review_manifest = {
        "schema_version": "vasu_instruction_quality_review_manifest_v1",
        "review_path": str(review_path),
        "review_sha256": report["review_sha256"],
        "decisions": 0,
        "status_counts": {"unreviewed": 500},
        "training_authorized": False,
    }
    atomic_write_text(
        Path(config["review_manifest_path"]),
        json.dumps(review_manifest, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(Path(config["review_packet_path"]), _review_packet(records, report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--artifacts-only",
        action="store_true",
        help="Verify the deterministic source and refresh reports without rewriting it.",
    )
    args = parser.parse_args()
    report = build_artifacts(args.config, regenerate_source=not args.artifacts_only)
    print(f"Source: {report['source_path']}")
    print(f"Source SHA-256: {report['source_sha256']}")
    print(f"Total records: {report['total_records']}")
    print(f"Category counts: {report['category_counts']}")
    print(f"Difficulty counts: {report['difficulty_counts']}")
    print(f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}")
    print(f"Near-duplicate candidates: {len(report['near_duplicate_candidates'])}")
    print(f"Truncated examples: {len(report['token_length_audit']['truncated_example_ids'])}")
    print("Automatic approval: False")
    print("Training authorized: False")


if __name__ == "__main__":
    main()
