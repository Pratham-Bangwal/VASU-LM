"""Run resumable full verified-arithmetic-v2 evaluation on a VASU checkpoint."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch

from evaluation.verified_arithmetic import (
    EVALUATOR_VERSION,
    evaluate_records,
    generate_answer,
    load_verified_split,
    sha256_file,
    summarize_results,
)
from vasu.config import get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer


DEFAULT_MANIFEST = Path(
    "data/processed/capability/verified_arithmetic_v2/manifest.json"
)
DEFAULT_TOKENIZER = Path("assets/tokenizer.json")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, records, split_path = load_verified_split(args.manifest, args.split)
    tokenizer_sha = sha256_file(args.tokenizer)
    if tokenizer_sha != manifest["tokenizer"]["sha256"]:
        raise ValueError("tokenizer SHA-256 differs from arithmetic manifest")
    checkpoint_sha = sha256_file(args.checkpoint)
    settings = {
        "mode": "greedy",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "prompt_format": "plain",
        "serialization": "Question: {prompt}\\nAnswer:",
    }
    identity = {
        "evaluator_version": EVALUATOR_VERSION,
        "split": args.split,
        "split_path": split_path.as_posix(),
        "split_sha256": sha256_file(split_path),
        "manifest_path": args.manifest.as_posix(),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint_path": args.checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer_path": args.tokenizer.as_posix(),
        "tokenizer_sha256": tokenizer_sha,
        "generation": settings,
    }
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = args.output_dir
    manifest_path = output / "run_manifest.json"
    results_path = output / "per_example.jsonl"
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume or a new output directory")
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity_sha256") != identity_hash:
            raise ValueError("evaluation resume identity mismatch")
    else:
        output.mkdir(parents=True)
        _atomic_json(
            manifest_path,
            {
                **identity,
                "identity_sha256": identity_hash,
                "git_commit": _git_commit(),
                "device": args.device,
                "created_at": datetime.now(UTC).isoformat(),
                "status": "in_progress",
            },
        )

    existing = _load_existing(results_path)
    completed = {str(item["id"]) for item in existing}
    if len(completed) != len(existing):
        raise ValueError("evaluation output contains duplicate IDs")
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    model = VASUModel(get_vasu_60m_config())
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    tokenizer = VASUTokenizer()
    tokenizer.load(str(args.tokenizer))
    combined = list(existing)
    for record in records:
        record_id = str(record["id"])
        if record_id in completed:
            continue
        result = evaluate_records(
            [record],
            lambda item: generate_answer(
                model,
                tokenizer,
                item,
                device,
                max_new_tokens=args.max_new_tokens,
            ),
        )
        combined.extend(result)
        completed.add(record_id)
        # Persist each completed example atomically. An interrupted evaluation
        # therefore resumes at the exact next logical record without duplicate
        # generation or relying on an in-memory-only progress counter.
        _atomic_jsonl(results_path, combined)
    summary = summarize_results(combined)
    _atomic_json(output / "category_summary.json", summary)
    lines = [
        f"Verified arithmetic v2 {args.split}",
        f"Checkpoint: {args.checkpoint}",
        f"Checkpoint SHA-256: {checkpoint_sha}",
        f"Records: {summary['overall']['count']}",
        f"Exact accuracy: {summary['overall']['exact_accuracy']}",
        f"Malformed: {summary['overall']['malformed']}",
        f"Unanswered: {summary['overall']['unanswered']}",
        f"Truncated: {summary['overall']['truncated']}",
        f"Prompt leakage: {summary['overall']['prompt_leakage']}",
        f"Duration seconds: {summary['overall']['duration_seconds']}",
        f"Generated tokens: {summary['overall']['generated_tokens']}",
        f"Throughput tok/s: {summary['overall']['throughput_tokens_per_second']}",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest["status"] = "complete"
    run_manifest["completed_at"] = datetime.now(UTC).isoformat()
    run_manifest["records"] = len(combined)
    _atomic_json(manifest_path, run_manifest)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "eval"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
