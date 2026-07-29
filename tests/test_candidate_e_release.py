from pathlib import Path

import pytest

from vasu.data.arithmetic_v2 import generate_records
from vasu.data.candidate_e_release import build_paired_release, validate_paired_release


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
