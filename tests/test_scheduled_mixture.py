"""Tests for deterministic, mixed-mask scheduled training data."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.build_vasu_capability_schedules import (
    CANDIDATES,
    CANDIDATES_V2,
    TOTAL_RECORDS,
    candidate_sources,
)
from vasu.data.scheduled_mixture import (
    ReplayOverride,
    ReplaySafetyPolicy,
    SCHEDULE_DTYPE,
    ScheduledPretrainingDataset,
    ScheduledSource,
    allocate_records,
    build_schedule,
    sha256_file,
    validate_schedule_release,
    validate_replay_policy,
    validate_sources,
    write_schedule_release,
)
from vasu.training.capability_cpt import (
    BLOCKED_MESSAGE,
    calculate_step_accounting,
    require_training_authorization,
    validate_capability_config,
)
from vasu.training.losses import language_model_loss
from vasu.training.trainer import Trainer


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


def test_replay_policy_warns_rejects_and_supports_scoped_override(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, "tiny", weight=1.0, order=0, records=2)
    warning = ReplaySafetyPolicy(warning_threshold=1.5, hard_limit=3.0)
    _, accounting = build_schedule(
        [source],
        4,
        42,
        replay_policy=warning,
    )
    assert accounting[0].replay_safety_status == "warning"
    assert accounting[0].maximum_reuse_count == 2
    assert accounting[0].reuse_count_distribution == {"2": 2}

    with pytest.raises(ValueError, match="replay hard limit exceeded"):
        build_schedule(
            [source],
            7,
            42,
            replay_policy=warning,
        )

    override = ReplaySafetyPolicy(
        warning_threshold=1.5,
        hard_limit=3.0,
        overrides=(
            ReplayOverride(
                source_id="tiny",
                approved_maximum_passes=4.0,
                justification="Bounded experiment-specific test approval.",
            ),
        ),
    )
    _, accounting = build_schedule(
        [source],
        7,
        42,
        replay_policy=override,
    )
    assert accounting[0].replay_safety_status == "override"
    assert accounting[0].replay_override is not None


def test_replay_override_requires_justification() -> None:
    policy = ReplaySafetyPolicy(
        overrides=(
            ReplayOverride(
                source_id="arithmetic",
                approved_maximum_passes=11.0,
                justification=" ",
            ),
        )
    )
    with pytest.raises(ValueError, match="written justification"):
        validate_replay_policy(policy)


@pytest.mark.parametrize("key", ["a", "b"])
def test_v1_candidate_arithmetic_fails_replay_gate(key: str) -> None:
    with pytest.raises(ValueError, match="verified_arithmetic_v1"):
        build_schedule(
            candidate_sources(CANDIDATES[key]),
            TOTAL_RECORDS,
            42,
            replay_policy=ReplaySafetyPolicy(),
        )


@pytest.mark.parametrize(
    ("key", "passes"),
    [("a", 1.198098742716958), ("b", 3.5946028825513645)],
)
def test_v2_candidates_pass_replay_gate(key: str, passes: float) -> None:
    _, accounting = build_schedule(
        candidate_sources(CANDIDATES_V2[key]),
        TOTAL_RECORDS,
        42,
        replay_policy=ReplaySafetyPolicy(),
    )
    arithmetic = next(
        item for item in accounting if item.source_id == "verified_arithmetic_v2"
    )
    assert arithmetic.effective_source_passes == pytest.approx(passes)
    assert arithmetic.replay_safety_status == "pass"


def test_v2_control_is_arithmetic_free_and_valid() -> None:
    sources = candidate_sources(CANDIDATES_V2["c"])
    assert all("arithmetic" not in source.identifier for source in sources)
    _, accounting = build_schedule(
        sources,
        TOTAL_RECORDS,
        42,
        replay_policy=ReplaySafetyPolicy(),
    )
    assert all(item.replay_safety_status == "pass" for item in accounting)


def test_candidate_c_authorized_and_candidates_a_b_remain_unauthorized() -> None:
    candidate_c = validate_capability_config(
        Path("configs/training/capability_cpt_c_control_20m_v2.json")
    )
    candidate_c["config"]["_authorization_config_path"] = (
        "configs/training/capability_cpt_c_control_20m_v2.json"
    )
    require_training_authorization(candidate_c["config"])

    for name in (
        "capability_cpt_a_factual_20m_v2.json",
        "capability_cpt_b_balanced_20m_v2.json",
    ):
        result = validate_capability_config(Path("configs/training") / name)
        assert result["config"]["technical_gates"] == {
            "replay_safety": "passed",
            "cuda_smoke": "passed",
            "cuda_exact_resume": "passed",
        }
        with pytest.raises(PermissionError, match=BLOCKED_MESSAGE):
            require_training_authorization(result["config"])


def test_authorization_template_is_not_authorizing() -> None:
    template = json.loads(
        Path(
            "configs/authorization/capability_cpt_c_control_20m_v2.authorization.template.json"
        ).read_text(encoding="utf-8")
    )
    assert template["status"] != "approved"
    assert template["decision"] != "authorized"


def test_true_authorization_still_requires_complete_technical_gates() -> None:
    with pytest.raises(PermissionError, match="cuda_exact_resume"):
        require_training_authorization(
            {
                "training_authorized": True,
                "technical_gates": {
                    "replay_safety": "passed",
                    "cuda_smoke": "passed",
                    "cuda_exact_resume": "not_completed",
                },
            }
        )


class _IndexedRecordingDataset(
    torch.utils.data.Dataset[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
):
    def __init__(
        self,
        base: ScheduledPretrainingDataset,
        indices: list[int],
        seen: list[tuple[str, int]],
    ) -> None:
        self.base = base
        self.indices = indices
        self.seen = seen

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_index = self.indices[index]
        self.seen.append(self.base.source_identity(source_index))
        return self.base[source_index]

    def resume_identity(self) -> dict[str, object]:
        identity = self.base.resume_identity()
        identity["selection_sha256"] = hashlib.sha256(
            json.dumps(self.indices).encode()
        ).hexdigest()
        return identity


class _TinyCudaLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32_000, 24)
        self.projection = nn.Linear(24, 32_000, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.embedding(token_ids))


def _cuda_config(tmp_path: Path, name: str) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=1,
        gradient_accumulation_steps=4,
        grad_clip=1.0,
        epochs=1,
        use_amp=True,
        seed=42,
        shuffle=False,
        drop_last=True,
        save_every_steps=0,
        checkpoint_path=str(tmp_path / name),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        num_workers=0,
        persistent_workers=False,
        optimizer_backend="standard",
        learning_rate=1e-4,
        weight_decay=0.01,
    )


def _nested_close(left: object, right: object) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        torch.testing.assert_close(
            left.detach().cpu(),
            right.detach().cpu(),
            rtol=1e-6,
            atol=1e-7,
        )
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _nested_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _nested_close(first, second)
    else:
        assert left == right


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("boundary", ["optimizer", "mid", "source", "replay"])
def test_cuda_scheduled_exact_resume(
    tmp_path: Path,
    boundary: str,
) -> None:
    """Bounded real-schedule proof across accumulation/source/replay boundaries."""

    base = ScheduledPretrainingDataset.from_resolved_manifest(
        Path(
            "data/processed/capability/mixtures/"
            "capability_cpt_b_balanced_20m_v2/resolved_manifest.json"
        )
    )
    if boundary in {"optimizer", "mid"}:
        indices = list(range(8))
        interrupt_after = 4 if boundary == "optimizer" else 3
    elif boundary == "source":
        transition = next(
            index
            for index in range(1, len(base))
            if base.source_identity(index)[0]
            != base.source_identity(index - 1)[0]
        )
        indices = list(range(max(0, transition - 2), transition + 6))
        interrupt_after = min(2, transition)
    else:
        seen_arithmetic: dict[int, int] = {}
        first = second = -1
        for index in range(len(base)):
            source, local = base.source_identity(index)
            if source != "verified_arithmetic_v2":
                continue
            if local in seen_arithmetic:
                first, second = seen_arithmetic[local], index
                break
            seen_arithmetic[local] = index
        assert first >= 0 and second >= 0
        indices = [first, 0, second, 1, 2, 3, 4, 5]
        interrupt_after = 2

    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    uninterrupted_seen: list[tuple[str, int]] = []
    uninterrupted = Trainer(
        _TinyCudaLanguageModel(),
        tokenizer=None,
        train_dataset=_IndexedRecordingDataset(
            base, indices, uninterrupted_seen
        ),
        val_dataset=_IndexedRecordingDataset(base, indices[:2], []),
        config=_cuda_config(tmp_path, "uninterrupted.pt"),
        device=torch.device("cuda"),
    )
    uninterrupted.train_epoch()
    expected_parameters = [
        parameter.detach().clone()
        for parameter in uninterrupted.model.parameters()
    ]
    expected_optimizer = uninterrupted.optimizer.state_dict()
    expected_scheduler = uninterrupted.scheduler.state_dict()
    expected_scaler = uninterrupted.scaler.state_dict()

    torch.manual_seed(991)
    torch.cuda.manual_seed_all(991)
    resumed_seen: list[tuple[str, int]] = []
    config = _cuda_config(tmp_path, "resumed.pt")
    interrupted = Trainer(
        _TinyCudaLanguageModel(),
        tokenizer=None,
        train_dataset=_IndexedRecordingDataset(base, indices, resumed_seen),
        val_dataset=_IndexedRecordingDataset(base, indices[:2], []),
        config=config,
        device=torch.device("cuda"),
    )
    interrupted.train_epoch(max_microbatches=interrupt_after)
    interrupted.save_training_checkpoint(config.checkpoint_path)
    resumed = Trainer(
        _TinyCudaLanguageModel(),
        tokenizer=None,
        train_dataset=_IndexedRecordingDataset(base, indices, resumed_seen),
        val_dataset=_IndexedRecordingDataset(base, indices[:2], []),
        config=config,
        device=torch.device("cuda"),
    )
    resumed.train_epoch()

    assert resumed_seen == uninterrupted_seen
    assert resumed.global_step == uninterrupted.global_step == 2
    assert resumed._optimizer_steps_in_epoch == 2
    _nested_close(expected_optimizer, resumed.optimizer.state_dict())
    _nested_close(expected_scheduler, resumed.scheduler.state_dict())
    _nested_close(expected_scaler, resumed.scaler.state_dict())
    for expected, actual in zip(
        expected_parameters,
        resumed.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            expected,
            actual.detach(),
            rtol=1e-6,
            atol=1e-7,
        )
