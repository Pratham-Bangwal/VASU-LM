from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vasu.training import orchestration


def _checkpoint(path: Path, step: int, *, complete: bool = True) -> None:
    payload = {
        "epoch": 0,
        "global_step": step,
        "loss": 1.0,
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {"state": {0: {"value": torch.tensor([2.0])}}},
        "scheduler": {"last_epoch": step},
    }
    if not complete:
        payload.pop("optimizer")
    torch.save(payload, path)


@pytest.fixture
def tiny_model_validation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        orchestration,
        "_validate_vasu_60m_model_state",
        lambda state: None,
    )


def test_latest_valid_checkpoint_selection(tmp_path: Path, tiny_model_validation):
    _checkpoint(tmp_path / "block_final_step_20.pt", 20)
    _checkpoint(tmp_path / "step_30.pt", 30)
    result = orchestration.find_latest_valid_checkpoint(tmp_path)
    assert result.valid and result.global_step == 30


def test_invalid_checkpoint_rejected(tmp_path: Path, tiny_model_validation):
    path = tmp_path / "step_10.pt"
    _checkpoint(path, 10, complete=False)
    assert not orchestration.inspect_checkpoint(path).valid


def test_filename_step_disagreement_uses_internal_step(
    tmp_path: Path, tiny_model_validation
):
    _checkpoint(tmp_path / "step_999.pt", 10)
    _checkpoint(tmp_path / "step_20.pt", 20)
    result = orchestration.find_latest_valid_checkpoint(tmp_path)
    assert result.global_step == 20
    assert result.filename_step_matches is True


def test_block_final_wins_same_internal_step(tmp_path: Path, tiny_model_validation):
    _checkpoint(tmp_path / "step_20.pt", 20)
    _checkpoint(tmp_path / "thermal_stop_step_20.pt", 20)
    _checkpoint(tmp_path / "block_final_step_20.pt", 20)
    result = orchestration.find_latest_valid_checkpoint(tmp_path)
    assert result.checkpoint_kind == "block_final"


def test_current_step_equals_target():
    assert orchestration.validate_progress(200_000, 200_000, 200_000) == 0


def test_current_step_above_target():
    with pytest.raises(ValueError, match="above"):
        orchestration.validate_progress(200_001, 200_001, 200_000)


def test_normal_200_step_progress():
    assert orchestration.validate_progress(152_400, 152_600, 200_000) == 200


def test_final_partial_block_progress():
    assert orchestration.validate_progress(199_900, 200_000, 200_000) == 100


def test_no_progress_failure():
    with pytest.raises(ValueError, match="did not increase"):
        orchestration.validate_progress(100, 100, 200_000)


def test_excessive_step_jump_failure():
    with pytest.raises(ValueError, match="more than 200"):
        orchestration.validate_progress(100, 301, 200_000)


def test_thermal_stop_detection():
    parsed = orchestration.parse_runner_output(
        "Final global step: 152450\nThermal stop occurred: True\n"
    )
    assert parsed["thermal_stop"] is True
    assert not orchestration.should_launch_next_block(
        current_step=152_450,
        target_step=200_000,
        thermal_stop=True,
    )


def test_failed_process_exit_code():
    with pytest.raises(RuntimeError, match="status 3"):
        orchestration.validate_process_exit(3)


def test_missing_checkpoint_after_successful_exit(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        orchestration.find_latest_valid_checkpoint(tmp_path)


def test_disk_below_threshold():
    with pytest.raises(RuntimeError, match="below"):
        orchestration.require_free_disk(9.9, 10.0)


def test_conflicting_training_process_detection():
    conflicts = orchestration.conflicting_commands(
        [(1, "python train_vasu_60m_fineweb_blocks.py"), (2, "notepad")]
    )
    assert conflicts == [
        {"pid": 1, "command_line": "python train_vasu_60m_fineweb_blocks.py"}
    ]


def test_step_152200_milestone_requirement_is_representable(tmp_path: Path):
    required = tmp_path / "fineweb_step_152200.pt"
    assert not required.exists()
    _checkpoint(required, 152_200)
    assert required.exists()


def test_final_step_200000_milestone_creation(
    tmp_path: Path, tiny_model_validation
):
    source = tmp_path / "block_final_step_200000.pt"
    destination = tmp_path / "milestones" / "fineweb_step_200000.pt"
    _checkpoint(source, 200_000)
    result = orchestration.preserve_milestone(
        source, destination, expected_step=200_000
    )
    assert destination.exists()
    assert result["global_step"] == 200_000
    assert result["sha256"] == orchestration.inspect_checkpoint(source).sha256


def test_logs_append_correctly(tmp_path: Path):
    path = tmp_path / "summary.jsonl"
    orchestration.append_jsonl(path, {"final_step": 10})
    orchestration.append_jsonl(path, {"final_step": 20})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"final_step": 10}, {"final_step": 20}]


def test_optional_metrics_remain_null_when_absent():
    parsed = orchestration.parse_runner_output("Block completed: True\n")
    assert parsed["validation_loss"] is None
    assert parsed["maximum_temperature"] is None


def test_ctrl_c_prevents_another_block_launch():
    assert not orchestration.should_launch_next_block(
        current_step=152_400,
        target_step=200_000,
        interrupted=True,
    )


def test_historical_wrapper_retains_old_target():
    text = Path("run_vasu_60m_for_11_hours.ps1").read_text(encoding="utf-8")
    assert "$TargetGlobalStep = 152000" in text
