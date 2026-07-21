from __future__ import annotations

from pathlib import Path

import pytest

import train_vasu_60m_fineweb_blocks as blocks


def test_start_200000_advances_one_block_toward_213100():
    assert blocks.resolve_target_global_step(200_000) == 200_200
    assert blocks.resolve_target_global_step(200_000) - 200_000 == 200


def test_historical_start_150200_still_advances_one_block():
    assert blocks.resolve_target_global_step(150_200) == 150_400
    assert blocks.resolve_target_global_step(150_200) - 150_200 == 200


def test_partial_block_is_capped_at_hard_target():
    assert blocks.resolve_target_global_step(213_050) == 213_100


def test_start_at_target_requires_zero_additional_steps():
    target = blocks.resolve_target_global_step(213_100)
    assert target == 213_100
    assert target - 213_100 == 0


def test_start_above_target_fails_clearly():
    with pytest.raises(ValueError, match="beyond the hard target"):
        blocks.resolve_target_global_step(213_101)


def test_target_cannot_advance_past_213100():
    for starting_step in (
        212_700,
        212_900,
        213_000,
        213_099,
        213_100,
    ):
        assert blocks.resolve_target_global_step(starting_step) <= 213_100


def test_total_unseen_continuation_from_200000_is_exact():
    assert blocks.TARGET_GLOBAL_STEP - 200_000 == 13_100


def test_step_to_offset_mapping():
    tokens_per_step = (
        blocks.BATCH_SIZE
        * blocks.GRADIENT_ACCUMULATION_STEPS
        * 256
    )
    assert 150_000 * tokens_per_step == 1_228_800_000
    assert 150_200 * tokens_per_step == 1_230_438_400
    assert 152_000 * tokens_per_step == 1_245_184_000
    assert 1_245_891_252 - 152_000 * tokens_per_step == 707_252
    assert 200 * tokens_per_step == 1_638_400
    assert 213_100 * tokens_per_step == 1_745_715_200
    assert 1_745_891_730 - 213_100 * tokens_per_step == 176_530


def test_retention_never_touches_milestone_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint_dir = tmp_path / "fineweb_blocks"
    checkpoint_dir.mkdir()
    milestone = tmp_path / "milestones" / "fineweb_step_150000.pt"
    milestone.parent.mkdir()
    milestone.write_bytes(b"preserved")
    protected_block = checkpoint_dir / "block_final_step_150200.pt"
    protected_block.write_bytes(b"protected")
    for step in (10, 20, 30, 40):
        (checkpoint_dir / f"step_{step}.pt").write_bytes(b"test")
    for step in (150400, 150600, 150800):
        (checkpoint_dir / f"block_final_step_{step}.pt").write_bytes(b"test")

    monkeypatch.setattr(blocks, "CHECKPOINT_DIR", checkpoint_dir)
    blocks.apply_checkpoint_retention()

    assert milestone.read_bytes() == b"preserved"
    assert protected_block.read_bytes() == b"protected"
    assert len(list(checkpoint_dir.glob("step_*.pt"))) == 3
    assert len(list(checkpoint_dir.glob("block_final_step_*.pt"))) == 3
