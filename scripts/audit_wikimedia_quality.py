"""Scan a prepared Wikimedia JSONL for every known extraction-defect family."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from vasu.data.preparation import load_preparation_config
from vasu.data.preparation.diagnostics import DefectFinding, find_known_defects
from vasu.data.preparation.quality import assess_final_chunk


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _context(text: str, finding: DefectFinding, radius: int = 140) -> str:
    start = max(0, finding.start - radius)
    end = min(len(text), finding.end + radius)
    return " ".join(text[start:end].split())


def audit_dataset(input_path: Path, config_path: Path) -> dict[str, Any]:
    config = load_preparation_config(config_path)
    matches: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    disposition_counts: Counter[str] = Counter()
    total_chunks = 0
    unexplained = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            total_chunks += 1
            text = str(record["cleaned_text"])
            findings = find_known_defects(text)
            if not findings:
                continue
            quality = assess_final_chunk(
                text,
                token_count=int(record["token_count"]),
                minimum_tokens=config.minimum_chunk_tokens,
                maximum_list_like_line_ratio=config.maximum_list_like_line_ratio,
                minimum_prose_sentences_for_list_chunk=(
                    config.minimum_prose_sentences_for_list_chunk
                ),
                boundary_start_type=str(record["boundary_start_type"]),
                boundary_end_type=str(record["boundary_end_type"]),
                training_mode=True,
            )
            for finding in findings:
                # Overlapping defects can legitimately produce a different
                # first rejection reason; any automatic rejection explains a
                # reject-disposition diagnostic finding.
                explained = (
                    finding.disposition == "manual_review"
                    or quality.rejection_reason is not None
                )
                if not explained:
                    unexplained += 1
                family_counts[finding.family] += 1
                disposition_counts[finding.disposition] += 1
                matches.append(
                    {
                        "jsonl_line_number": line_number,
                        "chunk_id": record["chunk_id"],
                        "title": record["title"],
                        "family": finding.family,
                        "disposition": finding.disposition,
                        "expected_rejection_reason": finding.rejection_reason,
                        "current_quality_rejection": quality.rejection_reason,
                        "explained_by_policy": explained,
                        "context": _context(text, finding),
                    }
                )
    return {
        "format_version": "wikimedia_quality_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "total_chunks": total_chunks,
        "matched_chunks": len({item["chunk_id"] for item in matches}),
        "total_findings": len(matches),
        "family_counts": dict(sorted(family_counts.items())),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "unexplained_match_count": unexplained,
        "matches": matches,
    }


def _text_report(report: dict[str, Any]) -> str:
    lines = [
        "WIKIMEDIA GLOBAL QUALITY AUDIT",
        "=" * 80,
        f"Input: {report['input_path']}",
        f"SHA-256: {report['input_sha256']}",
        f"Chunks: {report['total_chunks']}",
        f"Matched chunks: {report['matched_chunks']}",
        f"Total findings: {report['total_findings']}",
        f"Unexplained findings: {report['unexplained_match_count']}",
        "",
        "Family counts:",
    ]
    lines.extend(
        f"- {family}: {count}" for family, count in report["family_counts"].items()
    )
    lines.extend(["", "Matches:"])
    for item in report["matches"]:
        lines.extend(
            [
                "-" * 80,
                f"{item['chunk_id']} | {item['title']} | {item['family']}",
                f"Disposition: {item['disposition']}",
                f"Current rejection: {item['current_quality_rejection']}",
                f"Explained: {item['explained_by_policy']}",
                f"Context: {item['context']}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/pretrain/factual/wikimedia_pilot/documents.jsonl"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/preparation/wikimedia_pilot.json"),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--require-zero-unexplained", action="store_true")
    args = parser.parse_args()
    report = audit_dataset(args.input, args.config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_text.write_text(_text_report(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "input_sha256", "total_chunks", "matched_chunks", "total_findings",
        "family_counts", "disposition_counts", "unexplained_match_count",
    )}, indent=2))
    if args.require_zero_unexplained and report["unexplained_match_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
