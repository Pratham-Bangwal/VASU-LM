from pathlib import Path

import pytest

from vasu.data.arithmetic_v2 import generate_records
from vasu.data.candidate_e_release import (
    build_paired_release,
    validate_matched_budget,
    validate_paired_release,
)


def _splits():
    return {
        "train": generate_records("train", 220, 42)[:6],
        "development": generate_records("development", 220, 43)[:3],
        "evaluation": generate_records("evaluation", 220, 44)[:3],
    }


def test_builds_and_validates_paired_immutable_release(tmp_path: Path) -> None:
    output = tmp_path / "candidate_e"
    result = build_paired_release(
        logical_splits=_splits(),
        tokenizer_path=Path("assets/tokenizer.json"),
        output_dir=output,
        created_at="2026-07-29T00:00:00+00:00",
    )
    assert result["training_authorized"] is False
    assert validate_paired_release(
        output, tokenizer_path=Path("assets/tokenizer.json")
    )["arms"]
    assert (output / "final_answer" / "train_loss_mask.bin").is_file()
    with pytest.raises(FileExistsError):
        build_paired_release(
            logical_splits=_splits(),
            tokenizer_path=Path("assets/tokenizer.json"),
            output_dir=output,
        )


def test_matched_budget_requires_equal_records_and_updates() -> None:
    assert (
        validate_matched_budget(
            control_scheduled_records=128,
            treatment_scheduled_records=128,
            sequence_length=256,
            microbatch_size=2,
            gradient_accumulation=16,
            optimizer_updates=4,
        )["processed_tokens_per_arm"]
        == 32_768
    )
    with pytest.raises(ValueError, match="scheduled record counts"):
        validate_matched_budget(
            control_scheduled_records=128,
            treatment_scheduled_records=127,
            sequence_length=256,
            microbatch_size=2,
            gradient_accumulation=16,
            optimizer_updates=4,
        )
