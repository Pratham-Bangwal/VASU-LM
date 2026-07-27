"""Build immutable, unauthorized Candidate-D control and treatment schedules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vasu.data.arithmetic_operation_schedule import build_operation_schedule, sha256_file
from vasu.data.scheduled_mixture import (
    ReplaySafetyPolicy,
    ScheduledSource,
    build_resolved_manifest,
    build_schedule,
    validate_schedule_release,
)


ROOT = Path("data/processed/capability")
TOKENIZER = Path("assets/tokenizer.json")
TOKENIZER_SHA256 = "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
FINEWEB_SHA256 = "d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e"
WIKIMEDIA_SHA256 = "fee0b88832fc569ce73393bc232e29a80cb981e31779ce4c42129d15a0be876f"
FINEWEB_MANIFEST_SHA256 = "d6f1d112fbd0135563c173feaf700162982718a6ced6abd537df0e749c267a00"
WIKIMEDIA_MANIFEST_SHA256 = "cec126ae4e804a4f42699f96625752199c73b0caafbc4bf8b6d3bc3f47560a81"
REPLAY = ReplaySafetyPolicy(warning_threshold=8.0, hard_limit=10.0)


def source_records(path: Path, offset: int = 0, stride: int = 256) -> int:
    values = path.stat().st_size // 2 - offset
    return (values - 257) // stride + 1


def sources(weights: list[float], arithmetic: bool) -> list[ScheduledSource]:
    result = [
        ScheduledSource("fineweb_replay_after_200k", "standard_token_stream", Path("data/processed/pretrain/fineweb_extension_500m.bin"), None, Path("data/processed/pretrain/fineweb_manifest.json"), FINEWEB_SHA256, None, FINEWEB_MANIFEST_SHA256, "train", 257, 256, "uint16", None, source_records(Path("data/processed/pretrain/fineweb_extension_500m.bin"), 392508748), weights[0], 0, "deterministic_permutation_epochs", False, 392508748, 256),
        ScheduledSource("wikimedia_factual_release_6aa10d73", "standard_token_stream", Path("data/processed/pretrain/factual/wikimedia_release_6aa10d73_train.bin"), None, Path("data/manifests/factual/wikimedia_release_6aa10d73_tokenized.json"), WIKIMEDIA_SHA256, None, WIKIMEDIA_MANIFEST_SHA256, "train", 257, 256, "uint16", None, source_records(Path("data/processed/pretrain/factual/wikimedia_release_6aa10d73_train.bin")), weights[1], 1, "deterministic_permutation_epochs", False, 0, 256),
    ]
    if arithmetic:
        view = ROOT / "verified_arithmetic_v2_operation_view_v1"
        manifest = json.loads((view / "manifest.json").read_text(encoding="utf-8"))
        artifacts = manifest["artifacts"]
        result.append(ScheduledSource("verified_arithmetic_v2_operation_view_v1", "packed_masked", view / "train_tokens.bin", view / "train_loss_mask.bin", view / "manifest.json", artifacts["train_tokens.bin"]["sha256"], artifacts["train_loss_mask.bin"]["sha256"], sha256_file(view / "manifest.json"), "train", 257, 256, "uint16", "uint8", int(manifest["packed_record_count"]), weights[2], 2, "deterministic_permutation_epochs", True, 0, 257))
    return result


def validation_sources() -> dict[str, object]:
    return {
        "fineweb_validation": {"manifest": "data/processed/pretrain/fineweb_manifest.json", "role": "fixed_original_validation"},
        "wikimedia_validation": {"manifest": "data/manifests/factual/wikimedia_release_6aa10d73_tokenized.json", "role": "validation"},
        "arithmetic_development": {"path": "data/processed/capability/verified_arithmetic_v2/dev.jsonl", "sha256": "e3130b9584d052bdf6848b0d394e8515d522fd5f682ab9b6c9795535b7a903d0", "role": "development"},
        "arithmetic_evaluation": {"path": "data/processed/capability/verified_arithmetic_v2/eval.jsonl", "sha256": "5d3fe19e950af146f1c345e6164e7bbd56618e38d8cc76df328e40eca8da6dd2", "role": "evaluation"},
    }


def write_release(candidate: str, plan: Path, output: Path, values: list[float], operation: Path | None = None) -> dict[str, object]:
    items = sources(values, operation is not None)
    schedule, accounting = build_schedule(items, 39072, 42, replay_policy=REPLAY)
    if operation is not None:
        op = np.fromfile(operation / "schedule.bin", dtype=np.dtype([("record", "<u8"), ("stage", "<u4")]))
        positions = np.flatnonzero(schedule["source"] == 2)
        if len(op) != len(positions):
            raise ValueError("operation and combined arithmetic record counts differ")
        schedule["record"][positions] = op["record"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "schedule.bin").write_bytes(schedule.tobytes())
    manifest = build_resolved_manifest(candidate_id=candidate, requested_plan_path=plan, sources=items, schedule=schedule, accounting=accounting, output_dir=output, seed=42, tokenizer_path=TOKENIZER, tokenizer_sha256=TOKENIZER_SHA256, validation_sources=validation_sources(), created_at="2026-07-27T00:00:00+00:00", replay_policy=REPLAY)
    if operation is not None:
        manifest["operation_schedule"] = {"path": (operation / "manifest.json").as_posix(), "manifest_sha256": sha256_file(operation / "manifest.json"), "schedule_sha256": sha256_file(operation / "schedule.bin")}
    (output / "resolved_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validate_schedule_release(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    control_id = "capability_cpt_d_control_10m_from_a_v1"
    treatment_id = "capability_cpt_d_arithmetic_10m_from_a_v1"
    control = ROOT / "mixtures" / control_id
    treatment = ROOT / "mixtures" / treatment_id
    op_dir = ROOT / "operation_schedules" / treatment_id
    if args.validate_only:
        result = {"control": validate_schedule_release(control), "treatment": validate_schedule_release(treatment)}
    else:
        weights = json.loads(Path("configs/data/mixtures/vasu_60m_capability_d_arithmetic_10m_from_a_v1.operation_weights.json").read_text(encoding="utf-8"))["weights"]
        curriculum = json.loads(Path("configs/data/mixtures/vasu_60m_capability_d_arithmetic_10m_from_a_v1.curriculum.json").read_text(encoding="utf-8"))
        build_operation_schedule(view_dir=ROOT / "verified_arithmetic_v2_operation_view_v1", output_dir=op_dir, total_records=23443, operation_weights=weights, seed=42, with_replacement=bool(curriculum["replacement"]), stages=curriculum["stages"], overwrite=True)
        result = {"control": write_release(control_id, Path("configs/data/mixtures/vasu_60m_capability_d_control_10m_from_a_v1.json"), control, [0.91, 0.09]), "treatment": write_release(treatment_id, Path("configs/data/mixtures/vasu_60m_capability_d_arithmetic_10m_from_a_v1.json"), treatment, [0.31, 0.09, 0.60], op_dir)}
    print(json.dumps({key: {"schedule": value["schedule"]["sha256"], "records": value["total_records"]} for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
