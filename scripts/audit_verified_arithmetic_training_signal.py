"""Read-only audit of arithmetic-v2 answer supervision and tokenization."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from vasu.tokenizer.tokenizer import VASUTokenizer


DEFAULT_RELEASE = Path("data/processed/capability/verified_arithmetic_v2")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _logical_examples(records_path: Path) -> list[dict]:
    examples = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        examples.extend(json.loads(line)["logical_examples"])
    return examples


def build_report(release: Path, tokenizer: VASUTokenizer) -> dict:
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = [json.loads(line) for line in (release / "train_records.jsonl").read_text(encoding="utf-8").splitlines()]
    examples = [item for row in provenance for item in row["logical_examples"]]
    answer_lengths = Counter()
    by_operation: dict[str, list[int]] = defaultdict(list)
    templates = Counter()
    intermediate = 0
    for example in examples:
        length = len(tokenizer.encode(example["answer"]))
        answer_lengths[length] += 1
        by_operation[example["operation"]].append(length)
        templates[example["template_id"]] += 1
        # Equality is itself a valid one-token comparison answer; a newline is
        # the unambiguous marker of a multi-step serialized answer.
        intermediate += int("\n" in example["answer"])
    masks = np.fromfile(release / "train_loss_mask.bin", dtype=np.uint8)
    tokens = np.fromfile(release / "train_tokens.bin", dtype=np.uint16)
    if len(masks) != len(tokens):
        raise ValueError("token and mask lengths differ")
    supervised = int(masks.sum())
    width = int(manifest["record_width"])
    final_answer_supervision = 0
    for row, mask_row in zip(provenance, masks.reshape(-1, width), strict=True):
        for span, example in zip(row["example_spans"], row["logical_examples"], strict=True):
            end = int(span["end"])
            answer_width = len(tokenizer.encode(example["answer"])) + 1  # EOS
            final_answer_supervision += int(mask_row[end - answer_width:end].sum())
    return {
        "format_version": "vasu_arithmetic_training_signal_audit_v1",
        "release": {"manifest_sha256": _sha256(manifest_path), "tokenizer_sha256": manifest["tokenizer"]["sha256"]},
        "examples": len(examples),
        "answer_token_length_histogram": dict(sorted(answer_lengths.items())),
        "answer_token_length_mean": sum(k * v for k, v in answer_lengths.items()) / len(examples),
        "answer_token_length_by_operation": {key: sum(values) / len(values) for key, values in sorted(by_operation.items())},
        "template_counts": dict(sorted(templates.items())),
        "intermediate_answer_examples": intermediate,
        "supervised_target_tokens": supervised,
        "final_answer_supervised_tokens": final_answer_supervision,
        "nonfinal_supervised_tokens": supervised - final_answer_supervision,
        "supervised_targets_per_example": supervised / len(examples),
        "finding": (
            "Final answers contain no serialized intermediate steps"
            if intermediate == 0
            else "Some final answers contain serialized intermediate steps"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--tokenizer", type=Path, default=Path("assets/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = VASUTokenizer()
    tokenizer.load(str(args.tokenizer))
    report = build_report(args.release, tokenizer)
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
