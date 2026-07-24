"""Build or validate unauthorized VASU capability-CPT schedule releases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vasu.data.scheduled_mixture import (
    ScheduledSource,
    validate_schedule_release,
    write_schedule_release,
)


TOKENIZER_PATH = Path("assets/tokenizer.json")
TOKENIZER_SHA256 = (
    "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
)
TOTAL_RECORDS = 78_144
CONTEXT_LENGTH = 256
RECORD_WIDTH = 257
FINEWEB_OFFSET = 392_508_748


@dataclass(frozen=True)
class Candidate:
    key: str
    candidate_id: str
    plan_path: Path
    output_dir: Path
    weights: tuple[float, ...]
    include_arithmetic: bool


CANDIDATES = {
    "a": Candidate(
        "a",
        "capability_cpt_a_factual_20m_v1",
        Path("configs/data/mixtures/vasu_60m_capability_a_factual_20m.json"),
        Path(
            "data/processed/capability/mixtures/"
            "capability_cpt_a_factual_20m_v1"
        ),
        (0.86, 0.09, 0.05),
        True,
    ),
    "b": Candidate(
        "b",
        "capability_cpt_b_balanced_20m_v1",
        Path("configs/data/mixtures/vasu_60m_capability_b_balanced_20m.json"),
        Path(
            "data/processed/capability/mixtures/"
            "capability_cpt_b_balanced_20m_v1"
        ),
        (0.76, 0.09, 0.15),
        True,
    ),
    "c": Candidate(
        "c",
        "capability_cpt_c_control_20m_v1",
        Path("configs/data/mixtures/vasu_60m_capability_c_control_20m.json"),
        Path(
            "data/processed/capability/mixtures/"
            "capability_cpt_c_control_20m_v1"
        ),
        (0.91, 0.09),
        False,
    ),
}


def _available_records(
    path: Path,
    *,
    token_offset: int,
    stride: int,
    record_width: int,
) -> int:
    token_count = path.stat().st_size // 2
    available = token_count - token_offset
    if available < record_width:
        raise ValueError(f"source has no complete records: {path}")
    return (available - record_width) // stride + 1


def _validate_plan(candidate: Candidate) -> None:
    plan = json.loads(candidate.plan_path.read_text(encoding="utf-8"))
    weights = tuple(float(item["weight"]) for item in plan["sources"])
    if weights != candidate.weights:
        raise ValueError(f"candidate {candidate.key} plan weights changed")
    arithmetic_sources = [
        item
        for item in plan["sources"]
        if item["domain"] == "mathematics"
    ]
    if candidate.include_arithmetic != bool(arithmetic_sources):
        raise ValueError(
            f"candidate {candidate.key} arithmetic inclusion is inconsistent"
        )


def candidate_sources(candidate: Candidate) -> list[ScheduledSource]:
    """Resolve current canonical source artifacts for one candidate."""

    _validate_plan(candidate)
    fine_path = Path("data/processed/pretrain/fineweb_extension_500m.bin")
    wiki_path = Path(
        "data/processed/pretrain/factual/"
        "wikimedia_release_6aa10d73_train.bin"
    )
    sources = [
        ScheduledSource(
            identifier="fineweb_replay_after_200k",
            source_kind="standard_token_stream",
            token_path=fine_path,
            mask_path=None,
            manifest_path=Path(
                "data/processed/pretrain/fineweb_manifest.json"
            ),
            token_sha256=(
                "d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e"
            ),
            mask_sha256=None,
            manifest_sha256=(
                "d6f1d112fbd0135563c173feaf700162982718a6ced6abd537df0e749c267a00"
            ),
            split_role="train",
            record_width=RECORD_WIDTH,
            context_length=CONTEXT_LENGTH,
            token_dtype="uint16",
            mask_dtype=None,
            available_records=_available_records(
                fine_path,
                token_offset=FINEWEB_OFFSET,
                stride=CONTEXT_LENGTH,
                record_width=RECORD_WIDTH,
            ),
            weight=candidate.weights[0],
            order=0,
            replay_policy="deterministic_permutation_epochs",
            masked=False,
            token_offset=FINEWEB_OFFSET,
            record_stride=CONTEXT_LENGTH,
        ),
        ScheduledSource(
            identifier="wikimedia_factual_release_6aa10d73",
            source_kind="standard_token_stream",
            token_path=wiki_path,
            mask_path=None,
            manifest_path=Path(
                "data/manifests/factual/"
                "wikimedia_release_6aa10d73_tokenized.json"
            ),
            token_sha256=(
                "fee0b88832fc569ce73393bc232e29a80cb981e31779ce4c42129d15a0be876f"
            ),
            mask_sha256=None,
            manifest_sha256=(
                "cec126ae4e804a4f42699f96625752199c73b0caafbc4bf8b6d3bc3f47560a81"
            ),
            split_role="train",
            record_width=RECORD_WIDTH,
            context_length=CONTEXT_LENGTH,
            token_dtype="uint16",
            mask_dtype=None,
            available_records=_available_records(
                wiki_path,
                token_offset=0,
                stride=CONTEXT_LENGTH,
                record_width=RECORD_WIDTH,
            ),
            weight=candidate.weights[1],
            order=1,
            replay_policy="deterministic_permutation_epochs",
            masked=False,
            token_offset=0,
            record_stride=CONTEXT_LENGTH,
        ),
    ]
    if candidate.include_arithmetic:
        arithmetic_path = Path(
            "data/processed/capability/verified_arithmetic_v1/train_tokens.bin"
        )
        sources.append(
            ScheduledSource(
                identifier="verified_arithmetic_v1",
                source_kind="packed_masked",
                token_path=arithmetic_path,
                mask_path=Path(
                    "data/processed/capability/verified_arithmetic_v1/"
                    "train_loss_mask.bin"
                ),
                manifest_path=Path(
                    "data/processed/capability/verified_arithmetic_v1/"
                    "manifest.json"
                ),
                token_sha256=(
                    "341e575dab8a001ac30155cf674c5bd4b25d687048114d36a4578673ac72d141"
                ),
                mask_sha256=(
                    "2ddf37b626e3d6c6b92a28f6a885b99c2d995524949fbca4db1424b1cf71789d"
                ),
                manifest_sha256=(
                    "85e749f9e08e3908a2bb0f77791e6b7a824ac47ea3d499f7a38c80d940b62fea"
                ),
                split_role="train",
                record_width=RECORD_WIDTH,
                context_length=CONTEXT_LENGTH,
                token_dtype="uint16",
                mask_dtype="uint8",
                available_records=9,
                weight=candidate.weights[2],
                order=2,
                replay_policy="deterministic_permutation_epochs",
                masked=True,
                token_offset=0,
                record_stride=RECORD_WIDTH,
            )
        )
    return sources


def validation_sources() -> dict[str, Any]:
    return {
        "fineweb_validation": {
            "manifest": "data/processed/pretrain/fineweb_manifest.json",
            "role": "fixed_original_validation",
        },
        "wikimedia_validation": {
            "manifest": (
                "data/manifests/factual/"
                "wikimedia_release_6aa10d73_tokenized.json"
            ),
            "role": "validation",
        },
        "arithmetic_development": {
            "path": (
                "data/processed/capability/verified_arithmetic_v1/dev.jsonl"
            ),
            "sha256": (
                "da988af5dca386fe98d4a07f87a2eff6b14ef867853b0918dcac12aa52df306c"
            ),
            "role": "development",
        },
        "arithmetic_evaluation": {
            "path": (
                "data/processed/capability/verified_arithmetic_v1/eval.jsonl"
            ),
            "sha256": (
                "f10a8bf0e668f02c0b70cabb279dfd98353cbc2d8a7cb495bea25a640d318ceb"
            ),
            "role": "evaluation",
        },
    }


def build_candidate(
    candidate: Candidate,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    return write_schedule_release(
        output_dir=candidate.output_dir,
        candidate_id=candidate.candidate_id,
        requested_plan_path=candidate.plan_path,
        sources=candidate_sources(candidate),
        total_records=TOTAL_RECORDS,
        seed=42,
        tokenizer_path=TOKENIZER_PATH,
        tokenizer_sha256=TOKENIZER_SHA256,
        validation_sources=validation_sources(),
        overwrite=overwrite,
        dry_run=dry_run,
        created_at=created_at,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        choices=("a", "b", "c", "all"),
        default="all",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.validate_only:
        raise SystemExit("--dry-run and --validate-only are mutually exclusive")
    keys = tuple(CANDIDATES) if args.candidate == "all" else (args.candidate,)
    results = {}
    for key in keys:
        candidate = CANDIDATES[key]
        if args.validate_only:
            result = validate_schedule_release(candidate.output_dir)
        else:
            result = build_candidate(
                candidate,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        results[key] = {
            "candidate_id": result["candidate_id"],
            "total_records": result["total_records"],
            "total_tokens": result["total_tokens"],
            "schedule_sha256": result["schedule"]["sha256"],
            "allocations": result["allocations"],
            "training_authorized": result["training_authorized"],
            "dry_run": args.dry_run,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
