from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import train_vasu_60m_alpaca_masked_v2 as v2
import train_vasu_60m_alpaca_masked_v3 as v3
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.orchestration import CheckpointInspection


def _valid_inspection(step: int = 200_000) -> CheckpointInspection:
    return CheckpointInspection(
        valid=True,
        path=str(v3.BASE_CHECKPOINT),
        global_step=step,
        model_config="vasu_60m",
        has_model_state=True,
        has_optimizer_state=True,
        has_scheduler_state=True,
        finite_tensors=True,
        sha256="a" * 64,
        filename_step=None,
        filename_step_matches=None,
        checkpoint_kind=None,
        error=None,
    )


def _write_array(path: Path, values: list[int], dtype) -> None:
    np.asarray(values, dtype=dtype).tofile(path)


def test_fresh_initialization_uses_fineweb_step_200000():
    assert v3.BASE_GLOBAL_STEP == 200_000
    assert v3.BASE_CHECKPOINT == Path(
        "checkpoints/vasu_60m/milestones/fineweb_step_200000.pt"
    )
    assert v3.initialization_label(None) == (
        "Initialization source: FineWeb step-200000 base"
    )


def test_v3_output_directory_is_isolated():
    v3.validate_output_isolation()
    assert v3.CHECKPOINT_DIR == Path(
        "checkpoints/vasu_60m/alpaca_masked_v3_from_200k"
    )


def test_v2_output_directory_remains_a_protected_separate_path():
    assert Path("checkpoints/vasu_60m/alpaca_masked_v2") in (
        v3.PROTECTED_DIRECTORIES
    )
    assert v3.CHECKPOINT_DIR != Path(
        "checkpoints/vasu_60m/alpaca_masked_v2"
    )


def test_fineweb_milestone_is_never_an_output_path():
    assert v3.BASE_CHECKPOINT.parent.resolve() != v3.CHECKPOINT_DIR.resolve()
    with pytest.raises(ValueError, match="outside v3 output"):
        v3.write_v3_checkpoint_sidecar(v3.BASE_CHECKPOINT, 200_000, 0)


def test_wrong_step_base_checkpoint_is_rejected(monkeypatch):
    monkeypatch.setattr(
        v3, "inspect_checkpoint", lambda *args, **kwargs: _valid_inspection(200_400)
    )
    with pytest.raises(RuntimeError, match="global_step must be 200000"):
        v3.validate_base_checkpoint()


def test_invalid_base_checkpoint_is_rejected(monkeypatch):
    invalid = replace(_valid_inspection(), valid=False, error="truncated")
    monkeypatch.setattr(
        v3, "inspect_checkpoint", lambda *args, **kwargs: invalid
    )
    with pytest.raises(RuntimeError, match="invalid FineWeb base"):
        v3.validate_base_checkpoint()


def test_vasu_60m_configuration_mismatch_is_rejected(monkeypatch):
    mismatch = replace(_valid_inspection(), model_config="vasu_31m")
    monkeypatch.setattr(
        v3, "inspect_checkpoint", lambda *args, **kwargs: mismatch
    )
    with pytest.raises(RuntimeError, match="not a VASU-60M"):
        v3.validate_base_checkpoint()


def test_tokenizer_vocab_mismatch_is_rejected(tmp_path, monkeypatch):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    class FakeBackend:
        def get_vocab_size(self):
            return 31_999

        def token_to_id(self, token):
            return {"[PAD]": 0, "[EOS]": 3}[token]

    class FakeTokenizer:
        def __init__(self):
            self.tokenizer = FakeBackend()

        def load(self, path):
            assert Path(path) == tokenizer_path

    monkeypatch.setattr(v3, "VASUTokenizer", FakeTokenizer)
    metadata = {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_vocab_size": 32_000,
        "pad_token_id": 0,
        "eos_token_id": 3,
    }
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        v3.validate_tokenizer(metadata, tokenizer_path)


def test_token_and_mask_length_mismatch_is_rejected(tmp_path):
    data = tmp_path / "tokens.bin"
    mask = tmp_path / "mask.bin"
    _write_array(data, [1, 2, 3], np.uint16)
    _write_array(mask, [0, 1], np.uint8)
    metadata = {"record_length": 3}
    with pytest.raises(ValueError, match="different lengths"):
        v3.validate_dataset(metadata, "hash", data, mask)


def test_v3_resume_search_is_redirected_only_to_v3(monkeypatch):
    observed = []

    def fake_find():
        observed.append(v2.CHECKPOINT_DIR)
        return None

    monkeypatch.setattr(v2, "find_latest_checkpoint", fake_find)
    assert v3.find_v3_resume() is None
    assert observed == [v3.CHECKPOINT_DIR]


def test_existing_v3_checkpoint_takes_precedence(monkeypatch):
    checkpoint = (v3.CHECKPOINT_DIR / "step_200100.pt", {"global_step": 200_100})
    monkeypatch.setattr(v2, "find_latest_checkpoint", lambda: checkpoint)
    assert v3.find_v3_resume() == checkpoint
    assert v3.initialization_label(checkpoint) == (
        "Resume source: Alpaca masked v3 checkpoint"
    )


def test_masked_loss_alignment_is_identical_to_v2(tmp_path):
    data = tmp_path / "tokens.bin"
    mask = tmp_path / "mask.bin"
    _write_array(data, [10, 11, 12, 3, 0], np.uint16)
    _write_array(mask, [0, 0, 1, 1, 0], np.uint8)
    dataset = PackedInstructionDataset(str(data), str(mask), seq_len=4)
    x, y, target_mask = dataset[0]
    assert x.tolist() == [10, 11, 12, 3]
    assert y.tolist() == [11, 12, 3, 0]
    assert target_mask.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_assistant_eos_target_remains_supervised(tmp_path):
    data = tmp_path / "tokens.bin"
    mask = tmp_path / "mask.bin"
    _write_array(data, [10, 11, 12, 3, 0], np.uint16)
    _write_array(mask, [0, 0, 1, 1, 0], np.uint8)
    _, y, target_mask = PackedInstructionDataset(
        str(data), str(mask), seq_len=4
    )[0]
    eos_position = y.tolist().index(3)
    assert target_mask[eos_position].item() == 1.0


def test_prompt_targets_remain_ignored(tmp_path):
    data = tmp_path / "tokens.bin"
    mask = tmp_path / "mask.bin"
    _write_array(data, [10, 11, 12, 3, 0], np.uint16)
    _write_array(mask, [0, 0, 1, 1, 0], np.uint8)
    _, y, target_mask = PackedInstructionDataset(
        str(data), str(mask), seq_len=4
    )[0]
    assert y[0].item() == 11
    assert target_mask[0].item() == 0.0


def test_model_state_loading_is_strict():
    model = torch.nn.Linear(2, 2)
    incomplete = {"weight": torch.zeros_like(model.weight)}
    with pytest.raises(RuntimeError, match="Missing key"):
        v3.strict_load_model_state(model, incomplete)


def test_dry_run_output_snapshot_detects_no_writes(tmp_path):
    output = tmp_path / "v3"
    before = v3.output_snapshot(output)
    assert before == set()
    assert v3.output_snapshot(output) == before


def test_v3_hyperparameters_match_v2_exactly():
    names = (
        "EPOCHS",
        "BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "GRAD_CLIP",
        "USE_AMP",
        "SEED",
        "SAVE_EVERY_STEPS",
    )
    for name in names:
        assert getattr(v3, name) == getattr(v2, name)


def test_v3_reuses_v2_dataset_and_mask_unchanged():
    assert v3.DATA_FILE == v2.DATA_FILE
    assert v3.MASK_FILE == v2.MASK_FILE
    assert v3.DATASET_METADATA_FILE == v2.DATASET_METADATA_FILE
