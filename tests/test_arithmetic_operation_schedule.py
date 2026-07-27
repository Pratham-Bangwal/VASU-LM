from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

import vasu.data.arithmetic_operation_schedule as schedule


OPS = ("addition", "subtraction")


def _weights() -> dict[str, float]:
    return {"addition": 0.5, "subtraction": 0.5}


def _view(tmp_path: Path) -> Path:
    view = tmp_path / "view"
    view.mkdir(parents=True, exist_ok=True)
    manifest = view / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    rows = []
    for index in range(12):
        operation = OPS[index % len(OPS)]
        rows.append({"record_index": index, "operation": operation})
    (view / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return view


def _valid_operation_view(tmp_path: Path) -> Path:
    """A minimal, self-contained authoritative source and derived view."""
    source = tmp_path / "source.jsonl"
    example = {
        "id": "add-1",
        "split": "train",
        "operation": "addition",
        "difficulty_tier": "tier_1",
        "answer_type": "integer",
        "template_family": "direct",
        "text": "1+1=2",
    }
    source.write_text(
        json.dumps({"example_spans": [{"start": 0, "end": 3}], "logical_examples": [example]}) + "\n",
        encoding="utf-8",
    )
    view = tmp_path / "derived"
    view.mkdir()
    tokens = np.zeros((1, schedule.RECORD_WIDTH), dtype=np.uint16)
    tokens[0, :3] = [5, 6, 2]
    mask = np.zeros((1, schedule.RECORD_WIDTH), dtype=np.uint8)
    mask[0, 1:3] = 1
    (view / "train_tokens.bin").write_bytes(tokens.tobytes())
    (view / "train_loss_mask.bin").write_bytes(mask.tobytes())
    row = {
        "record_index": 0,
        "operation": "addition",
        "difficulty_tier": "tier_1",
        "source_example_ids": ["add-1"],
        "source_examples": [example],
    }
    (view / "records.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (view / "source_operation_index.jsonl").write_text("{}\n", encoding="utf-8")
    artifacts = {}
    for name in ("train_tokens.bin", "train_loss_mask.bin", "records.jsonl", "source_operation_index.jsonl"):
        path = view / name
        artifacts[name] = {"path": name, "sha256": schedule.sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": schedule.VIEW_FORMAT,
        "training_authorized": False,
        "source": {"train_records_path": source.as_posix(), "train_records_sha256": schedule.sha256_file(source)},
        "artifacts": artifacts,
        "packed_record_count": 1,
    }
    (view / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return view


def _build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: object) -> dict[str, object]:
    view = _view(tmp_path)
    monkeypatch.setattr(schedule, "validate_operation_view", lambda _: {})
    total_records = int(kwargs.pop("total_records", 10))
    seed = int(kwargs.pop("seed", 42))
    with_replacement = bool(kwargs.pop("with_replacement", True))
    operation_weights = kwargs.pop("operation_weights", _weights())
    return schedule.build_operation_schedule(
        view_dir=view,
        output_dir=tmp_path / "schedule",
        total_records=total_records,
        operation_weights=operation_weights,
        seed=seed,
        with_replacement=with_replacement,
        **kwargs,
    )


def test_identical_seed_produces_identical_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = _build(monkeypatch, tmp_path / "one")
    second = _build(monkeypatch, tmp_path / "two")
    assert first["schedule"]["sha256"] == second["schedule"]["sha256"]


def test_different_seed_changes_schedule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = _build(monkeypatch, tmp_path / "one")
    second = _build(monkeypatch, tmp_path / "two", seed=43)
    assert first["schedule"]["sha256"] != second["schedule"]["sha256"]


def test_exact_record_and_token_accounting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _build(monkeypatch, tmp_path)
    assert result["total_records"] == 10
    assert result["total_tokens"] == 2560
    assert result["operation_counts"] == {"addition": 5, "subtraction": 5}


@pytest.mark.parametrize("weights", [{"addition": 0.2}, {"addition": -1.0, "subtraction": 2.0}, {"unknown": 1.0}, {"addition": 0.0, "subtraction": 0.0}])
def test_invalid_weights_rejected(weights: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        schedule._largest_remainder(10, weights)


def test_rounding_tie_break_is_stable() -> None:
    assert schedule._largest_remainder(1, {"addition": 0.5, "subtraction": 0.5}) == {"addition": 1, "subtraction": 0}


def test_replacement_duplicate_accounting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _build(monkeypatch, tmp_path, total_records=20)
    assert result["duplicate_records"] == 8


def test_no_replacement_and_impossible_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _build(monkeypatch, tmp_path / "ok", with_replacement=False, total_records=10)
    assert result["duplicate_records"] == 0
    with pytest.raises(ValueError, match="no-replacement"):
        _build(monkeypatch, tmp_path / "bad", with_replacement=False, total_records=20)


def test_curriculum_stage_assignment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stages = (
        {"id": "first", "weight": 0.5, "operations": ["addition"]},
        {"id": "second", "weight": 0.5, "operations": ["subtraction"]},
    )
    result = _build(monkeypatch, tmp_path, stages=stages)
    assert result["stage_counts"] == {"0": 5, "1": 5}


def test_schedule_rejects_stale_view_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _build(monkeypatch, tmp_path)
    manifest = tmp_path / "schedule" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["view_manifest_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source-view"):
        schedule.validate_operation_schedule(tmp_path / "schedule")
    assert result["schedule"]["sha256"]


def test_operation_index_rejects_non_train_examples(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    payload = {"example_spans": [{"start": 0, "end": 2}], "logical_examples": [{"id": "bad", "split": "evaluation", "operation": "addition", "difficulty_tier": "tier_1", "answer_type": "integer", "template_family": "direct"}]}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="train"):
        schedule.build_operation_index(path)


def test_operation_view_accepts_aligned_eos_supervision(tmp_path: Path) -> None:
    view = _valid_operation_view(tmp_path)
    assert schedule.validate_operation_view(view)["packed_record_count"] == 1


def test_operation_view_rejects_supervised_padding_target(tmp_path: Path) -> None:
    view = _valid_operation_view(tmp_path)
    mask = np.fromfile(view / "train_loss_mask.bin", dtype=np.uint8)
    mask[3] = 1
    (view / "train_loss_mask.bin").write_bytes(mask.tobytes())
    payload = json.loads((view / "manifest.json").read_text(encoding="utf-8"))
    payload["artifacts"]["train_loss_mask.bin"]["sha256"] = schedule.sha256_file(view / "train_loss_mask.bin")
    (view / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="PAD target"):
        schedule.validate_operation_view(view)


def test_operation_view_rejects_mixed_operation_provenance(tmp_path: Path) -> None:
    view = _valid_operation_view(tmp_path)
    row = json.loads((view / "records.jsonl").read_text(encoding="utf-8"))
    row["source_examples"][0]["operation"] = "subtraction"
    (view / "records.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    payload = json.loads((view / "manifest.json").read_text(encoding="utf-8"))
    payload["artifacts"]["records.jsonl"]["sha256"] = schedule.sha256_file(view / "records.jsonl")
    (view / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid provenance"):
        schedule.validate_operation_view(view)


def test_operation_view_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    view = _valid_operation_view(tmp_path)
    (tmp_path / "source.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash"):
        schedule.validate_operation_view(view)


def test_operation_view_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    view = _valid_operation_view(tmp_path)
    (view / "records.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        schedule.validate_operation_view(view)


def test_schedule_hash_matches_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _build(monkeypatch, tmp_path)
    path = tmp_path / "schedule" / "schedule.bin"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["schedule"]["sha256"]


def test_schedule_entries_record_their_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stages = (
        {"id": "first", "weight": 0.5, "operations": ["addition"]},
        {"id": "second", "weight": 0.5, "operations": ["subtraction"]},
    )
    _build(monkeypatch, tmp_path, stages=stages)
    entries = np.fromfile(tmp_path / "schedule" / "schedule.bin", dtype=schedule.SCHEDULE_DTYPE)
    assert entries["stage"].tolist() == [0] * 5 + [1] * 5


def test_stage_constraints_never_silently_place_an_operation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stages = (
        {"id": "addition", "weight": 0.5, "operations": ["addition"]},
        {"id": "subtraction", "weight": 0.5, "operations": ["subtraction"]},
    )
    with pytest.raises(ValueError, match="stage constraints"):
        _build(
            monkeypatch,
            tmp_path,
            stages=stages,
            operation_weights={"addition": 0.9, "subtraction": 0.1},
        )


def test_global_rng_state_is_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    random.seed(8675309)
    expected = random.random()
    random.seed(8675309)
    _build(monkeypatch, tmp_path)
    assert random.random() == expected


def test_atomic_write_cleans_temporary_file_after_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    real_replace = schedule.os.replace
    monkeypatch.setattr(schedule.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(OSError, match="injected"):
        schedule._atomic_bytes(target, b"payload")
    assert not target.exists()
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))
    monkeypatch.setattr(schedule.os, "replace", real_replace)


def test_schedule_overwrite_replaces_existing_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = _build(monkeypatch, tmp_path, seed=42)
    second = _build(monkeypatch, tmp_path, seed=43, overwrite=True)
    assert first["schedule"]["sha256"] != second["schedule"]["sha256"]
    assert not list(tmp_path.glob(".schedule.*"))


def test_failed_schedule_replacement_preserves_existing_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original = _build(monkeypatch, tmp_path, seed=42)
    destination = tmp_path / "schedule"
    real_replace = schedule.os.replace

    def fail_staging_promotion(source: str | Path, target: str | Path) -> None:
        if Path(target) == destination and ".staging-" in Path(source).name:
            raise OSError("injected promotion failure")
        real_replace(source, target)

    monkeypatch.setattr(schedule.os, "replace", fail_staging_promotion)
    with pytest.raises(OSError, match="injected promotion failure"):
        _build(monkeypatch, tmp_path, seed=43, overwrite=True)
    assert schedule.validate_operation_schedule(destination)["schedule"]["sha256"] == original["schedule"]["sha256"]
    assert not list(tmp_path.glob(".schedule.*"))
