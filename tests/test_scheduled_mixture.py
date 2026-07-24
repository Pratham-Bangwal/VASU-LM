"""Tests for deterministic, mixed-mask scheduled training data."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from scripts.build_vasu_capability_schedules import (
    CANDIDATES,
    TOTAL_RECORDS,
    candidate_sources,
)
from vasu.data.scheduled_mixture import (
    SCHEDULE_DTYPE,
    ScheduledPretrainingDataset,
    ScheduledSource,
    allocate_records,
    build_schedule,
    sha256_file,
    validate_schedule_release,
    validate_sources,
    write_schedule_release,
)
from vasu.training.capability_cpt import (
    BLOCKED_MESSAGE,
    calculate_step_accounting,
    require_training_authorization,
)
from vasu.training.losses import language_model_loss


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source(
    root: Path,
    identifier: str,
    *,
    weight: float,
    order: int,
    masked: bool = False,
    records: int = 4,
) -> ScheduledSource:
    width = 5
    tokens = (
        np.arange(records * width, dtype=np.uint16).reshape(records, width)
        + order * 100
    )
    token_path = root / f"{identifier}.bin"
    tokens.tofile(token_path)
    mask_path = None
    mask_hash = None
    mask_dtype = None
    artifacts: dict[str, object] = {
        "train_tokens.bin": {"path": token_path.name}
    }
    if masked:
        masks = np.ones_like(tokens, dtype=np.uint8)
        masks[:, 0] = 0
        masks[:, -1] = 0
        mask_path = root / f"{identifier}.mask.bin"
        masks.tofile(mask_path)
        mask_hash = sha256_file(mask_path)
        mask_dtype = "uint8"
    manifest_path = root / f"{identifier}.json"
    _json(manifest_path, {"artifacts": artifacts})
    return ScheduledSource(
        identifier=identifier,
        source_kind="packed_masked" if masked else "standard_token_stream",
        token_path=token_path,
        mask_path=mask_path,
        manifest_path=manifest_path,
        token_sha256=sha256_file(token_path),
        mask_sha256=mask_hash,
        manifest_sha256=sha256_file(manifest_path),
        split_role="train",
        record_width=width,
        context_length=width - 1,
        token_dtype="uint16",
        mask_dtype=mask_dtype,
        available_records=records,
        weight=weight,
        order=order,
        replay_policy="deterministic_permutation_epochs",
        masked=masked,
        record_stride=width,
    )


def _release(tmp_path: Path) -> tuple[Path, list[ScheduledSource]]:
    plan = tmp_path / "plan.json"
    _json(plan, {"name": "test"})
    tokenizer = tmp_path / "tokenizer.json"
    _json(tokenizer, {"test": True})
    sources = [
        _source(tmp_path, "plain", weight=0.5, order=0),
        _source(tmp_path, "masked", weight=0.5, order=1, masked=True),
    ]
    output = tmp_path / "release"
    write_schedule_release(
        output_dir=output,
        candidate_id="test",
        requested_plan_path=plan,
        sources=sources,
        total_records=8,
        seed=42,
        tokenizer_path=tokenizer,
        tokenizer_sha256=sha256_file(tokenizer),
        validation_sources={"dev": {"role": "validation"}},
        created_at="2026-01-01T00:00:00+00:00",
    )
    return output, sources


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("a", [67_204, 7_033, 3_907]),
        ("b", [59_389, 7_033, 11_722]),
        ("c", [71_111, 7_033]),
    ],
)
def test_candidate_allocations_are_exact(key: str, expected: list[int]) -> None:
    sources = candidate_sources(CANDIDATES[key])
    assert allocate_records(TOTAL_RECORDS, sources) == expected
    assert sum(expected) == TOTAL_RECORDS
    assert ("verified_arithmetic_v1" in {s.identifier for s in sources}) == (
        key != "c"
    )


def test_largest_remainder_uses_explicit_order(tmp_path: Path) -> None:
    sources = [
        _source(tmp_path, "later", weight=0.5, order=1),
        _source(tmp_path, "earlier", weight=0.5, order=0),
    ]
    assert allocate_records(3, sources) == [1, 2]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda source: replace(source, split_role="validation"), "training split"),
        (lambda source: replace(source, token_dtype="uint32"), "uint16"),
        (lambda source: replace(source, record_width=7), "context length"),
        (lambda source: replace(source, weight=float("nan")), "finite"),
    ],
)
def test_source_validation_rejects_invalid_metadata(
    tmp_path: Path, mutation, message: str
) -> None:
    source = mutation(_source(tmp_path, "source", weight=1.0, order=0))
    with pytest.raises(ValueError, match=message):
        validate_sources([source], check_hashes=False)


def test_source_validation_rejects_duplicates_and_hash_mismatches(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, "source", weight=1.0, order=0)
    with pytest.raises(ValueError, match="duplicate"):
        validate_sources([source, source], check_hashes=False)
    with pytest.raises(ValueError, match="token hash mismatch"):
        validate_sources(
            [replace(source, token_sha256="0" * 64)],
            check_hashes=True,
        )
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        validate_sources(
            [replace(source, manifest_sha256="0" * 64)],
            check_hashes=True,
        )


def test_schedule_is_deterministic_and_replay_epochs_change(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, "tiny", weight=1.0, order=0, records=3)
    first, accounting = build_schedule([source], 10, 7)
    second, _ = build_schedule([source], 10, 7)
    assert first.tobytes() == second.tobytes()
    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(
        second.tobytes()
    ).digest()
    assert first["record"][:3].tolist() != first["record"][3:6].tolist()
    assert int(first["record"].max()) < 3
    assert accounting[0].replay_epochs == 4
    assert accounting[0].reused_records == 7


def test_release_and_dataset_preserve_masks_and_batch_contract(
    tmp_path: Path,
) -> None:
    output, _ = _release(tmp_path)
    manifest = validate_schedule_release(output)
    dataset = ScheduledPretrainingDataset.from_resolved_manifest(
        output / "resolved_manifest.json"
    )
    assert len(dataset) == 8
    source_to_index: dict[str, int] = {}
    for index in range(len(dataset)):
        source_to_index.setdefault(dataset.source_identity(index)[0], index)
    plain = dataset[source_to_index["plain"]]
    masked = dataset[source_to_index["masked"]]
    assert torch.all(plain[2] == 1)
    assert masked[2].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert torch.equal(plain[0][1:], plain[1][:-1])
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert len(batch) == 3
    assert batch[0].shape == batch[1].shape == batch[2].shape == (2, 4)
    assert manifest["total_tokens"] == 32


def test_masked_loss_equals_manual_and_ignores_masked_targets(
    tmp_path: Path,
) -> None:
    output, _ = _release(tmp_path)
    dataset = ScheduledPretrainingDataset.from_resolved_manifest(
        output / "resolved_manifest.json"
    )
    index = next(
        i for i in range(len(dataset)) if dataset.source_identity(i)[0] == "masked"
    )
    _, targets, mask = dataset[index]
    logits = torch.randn(1, 4, 256)
    loss = language_model_loss(logits, targets.unsqueeze(0), mask.unsqueeze(0))
    token_losses = torch.nn.functional.cross_entropy(
        logits.view(-1, 256),
        targets,
        reduction="none",
    )
    expected = (token_losses * mask).sum() / mask.sum()
    assert torch.allclose(loss, expected)


def test_dataset_is_pickle_safe_and_identity_is_hash_bound(tmp_path: Path) -> None:
    output, _ = _release(tmp_path)
    dataset = ScheduledPretrainingDataset.from_resolved_manifest(
        output / "resolved_manifest.json"
    )
    dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored.source_identity(0) == dataset.source_identity(0)
    assert all(torch.equal(a, b) for a, b in zip(restored[0], dataset[0], strict=True))
    assert restored.resume_identity()["schedule_sha256"] == sha256_file(
        output / "schedule.bin"
    )


def test_release_overwrite_and_corruption_protection(tmp_path: Path) -> None:
    output, _ = _release(tmp_path)
    original = (output / "schedule.bin").read_bytes()
    with pytest.raises(FileExistsError):
        _release(tmp_path)
    corrupted = bytearray(original)
    corrupted[0] ^= 1
    (output / "schedule.bin").write_bytes(corrupted)
    with pytest.raises(ValueError, match="schedule hash mismatch"):
        validate_schedule_release(output, check_source_hashes=False)


def test_step_accounting_and_authorization_gate() -> None:
    accounting = calculate_step_accounting(
        records=78_144,
        batch_size=2,
        accumulation_steps=16,
        sequence_length=256,
        warmup_fraction=0.02,
    )
    assert accounting.dataloader_microbatches == 39_072
    assert accounting.optimizer_updates == 2_442
    assert accounting.final_group_complete
    assert accounting.processed_tokens == 20_004_864
    assert accounting.warmup_updates == 49
    with pytest.raises(PermissionError, match=BLOCKED_MESSAGE):
        require_training_authorization({"training_authorized": False})


def test_schedule_dtype_is_compact_and_non_object() -> None:
    assert SCHEDULE_DTYPE.itemsize == 12
    assert not SCHEDULE_DTYPE.hasobject
