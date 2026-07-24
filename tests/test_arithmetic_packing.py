from __future__ import annotations

import numpy as np
import pytest
import torch

from vasu.data.arithmetic_packing import (
    DEFAULT_SEQUENCE_LENGTH,
    TokenizedArithmeticExample,
    pack_arithmetic_examples,
    pack_arithmetic_unique_pass,
    summarize_packed_records,
    unique_pass_record_count,
)
from vasu.training.instruction_dataset import PackedInstructionDataset
from vasu.training.losses import language_model_loss


EOS = 2
PAD = 3


def _example(identifier: str, length: int, start: int) -> TokenizedArithmeticExample:
    assert length >= 2
    return TokenizedArithmeticExample(identifier, tuple(range(start, start + length - 1)) + (EOS,))


def _exact_examples() -> list[TokenizedArithmeticExample]:
    return [
        _example("add", 100, 10),
        _example("subtract", 80, 200),
        _example("multiply", 77, 400),
    ]


def _pack(
    examples: list[TokenizedArithmeticExample],
    **kwargs: object,
) -> list:
    return pack_arithmetic_examples(
        examples,
        record_count=kwargs.pop("record_count", 1),
        eos_token_id=EOS,
        pad_token_id=PAD,
        seed=kwargs.pop("seed", 7),
        **kwargs,
    )


def test_one_and_multiple_examples_can_exactly_fill_a_record() -> None:
    one = _pack([_example("whole", 257, 10)])[0]
    multiple = _pack(_exact_examples())[0]
    for record in (one, multiple):
        assert record.used_token_count == 257
        assert record.padding_token_count == 0
        assert record.utilization_ratio == 1.0
        assert PAD not in record.tokens
        assert record.target_mask.shape == (DEFAULT_SEQUENCE_LENGTH,)


def test_non_exact_fill_has_explicit_tail_padding_and_complete_examples() -> None:
    first = _example("first", 120, 10)
    second = _example("second", 120, 200)
    record = _pack([first, second])[0]
    assert set(record.example_ids) == {"first", "second"}
    assert record.used_token_count == 240
    assert record.padding_token_count == 17
    assert np.all(record.tokens[240:] == PAD)
    assert PAD not in record.tokens[:240]
    assert all(
        tuple(record.tokens[span.start : span.end])
        == next(item.token_ids for item in (first, second) if item.source_id == span.source_id)
        for span in record.spans
    )


def test_boundaries_and_padding_are_never_supervised() -> None:
    record = _pack([_example("first", 120, 10), _example("second", 120, 200)])[0]
    left, right = record.spans
    # The EOS target remains supervised.
    assert record.tokens[left.end - 1] == EOS
    assert record.target_mask[left.end - 2] == 1
    # EOS_A -> first_B is masked.
    assert record.tokens[left.end - 1] == EOS
    assert record.tokens[left.end] == record.tokens[right.start]
    assert record.target_mask[left.end - 1] == 0
    # EOS_B -> PAD and PAD -> PAD are masked.
    assert record.tokens[right.end - 1] == EOS
    assert record.tokens[right.end] == PAD
    assert record.target_mask[right.end - 1] == 0
    assert np.all(record.target_mask[right.end:] == 0)
    targets = record.tokens[1:]
    inputs = record.tokens[:-1]
    assert np.all(record.target_mask[targets == PAD] == 0)
    assert np.all(record.target_mask[inputs == PAD] == 0)


def test_padding_only_appears_at_tail_and_no_example_is_split_or_truncated() -> None:
    examples = [_example("a", 150, 10), _example("b", 120, 200)]
    records = _pack(examples, record_count=2)
    assert {record.example_ids for record in records} == {("a",), ("b",)}
    for record in records:
        pad_positions = np.flatnonzero(record.tokens == PAD)
        if len(pad_positions):
            assert np.array_equal(pad_positions, np.arange(pad_positions[0], 257))
        for span in record.spans:
            expected = next(item.token_ids for item in examples if item.source_id == span.source_id)
            assert tuple(record.tokens[span.start : span.end]) == expected


def test_oversized_and_empty_sources_fail_without_pad_only_records() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _pack([])
    with pytest.raises(ValueError, match="exceeds"):
        _pack([_example("too_large", 258, 10)])


def test_determinism_replay_and_requested_record_count() -> None:
    examples = [_example("a", 150, 10), _example("b", 120, 200)]
    first = _pack(examples, record_count=5, seed=42)
    second = _pack(examples, record_count=5, seed=42)
    assert len(first) == len(second) == 5
    assert [record.tokens.tobytes() for record in first] == [
        record.tokens.tobytes() for record in second
    ]
    assert [record.target_mask.tobytes() for record in first] == [
        record.target_mask.tobytes() for record in second
    ]
    assert [record.spans for record in first] == [record.spans for record in second]
    assert [record.replay_epoch for record in first] == [0, 0, 1, 1, 2]
    assert first[0].example_ids != ()
    assert first[0].padding_token_count > 0


def test_unique_pass_emits_every_example_once_without_replay() -> None:
    examples = [
        _example("a", 150, 10),
        _example("b", 120, 200),
        _example("c", 80, 400),
    ]
    count = unique_pass_record_count(
        examples,
        eos_token_id=EOS,
        pad_token_id=PAD,
        seed=19,
    )
    records = pack_arithmetic_unique_pass(
        examples,
        eos_token_id=EOS,
        pad_token_id=PAD,
        seed=19,
    )
    consumed = [
        example_id for record in records for example_id in record.example_ids
    ]
    assert len(records) == count
    assert len(consumed) == len(examples)
    assert set(consumed) == {"a", "b", "c"}
    assert all(record.replay_epoch == 0 for record in records)


def test_utilization_statistics_and_optional_warning() -> None:
    records = _pack([_example("a", 150, 10), _example("b", 120, 200)], record_count=2)
    stats = summarize_packed_records(records)
    assert stats.packed_records == 2
    assert stats.total_logical_examples_consumed == 2
    assert stats.unique_examples_consumed == 2
    assert stats.replay_epochs == 1
    assert stats.real_tokens == 270
    assert stats.padding_tokens == 244
    assert stats.mean_utilization == pytest.approx(270 / 514)
    assert stats.minimum_utilization == pytest.approx(120 / 257)
    assert stats.maximum_utilization == pytest.approx(150 / 257)
    assert stats.padding_percentage == pytest.approx(244 / 514)
    with pytest.warns(RuntimeWarning, match="utilization"):
        _pack(
            [_example("a", 150, 10), _example("b", 120, 200)],
            record_count=2,
            utilization_warning_threshold=0.75,
        )
    with pytest.raises(ValueError, match="utilization"):
        _pack(
            [_example("a", 150, 10), _example("b", 120, 200)],
            record_count=2,
            utilization_warning_threshold=0.75,
            strict_utilization=True,
        )


def test_masked_loss_uses_only_enabled_targets() -> None:
    record = _pack([_example("first", 120, 10), _example("second", 120, 200)])[0]
    targets = torch.tensor(record.tokens[1:], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(record.target_mask, dtype=torch.float32).unsqueeze(0)
    vocab_size = 512
    logits = torch.full((1, DEFAULT_SEQUENCE_LENGTH, vocab_size), -20.0)
    logits.scatter_(2, targets.unsqueeze(-1), 20.0)
    # Make every masked boundary/padding target incorrect; CE stays zero.
    for position, enabled in enumerate(record.target_mask):
        if not enabled:
            logits[0, position, :] = 0.0
            logits[0, position, (targets[0, position] + 1) % vocab_size] = 20.0
    assert language_model_loss(logits, targets, mask).item() < 1e-5


def test_packed_dataset_and_record_index_resume_mapping(tmp_path) -> None:
    records = _pack([_example("a", 150, 10), _example("b", 120, 200)], record_count=4)
    token_path = tmp_path / "arithmetic_tokens.bin"
    mask_path = tmp_path / "arithmetic_mask.bin"
    np.concatenate([record.tokens for record in records]).tofile(token_path)
    np.concatenate([record.stored_mask for record in records]).tofile(mask_path)
    full = PackedInstructionDataset(str(token_path), str(mask_path), DEFAULT_SEQUENCE_LENGTH)
    resumed = PackedInstructionDataset(
        str(token_path), str(mask_path), DEFAULT_SEQUENCE_LENGTH, start_record=2
    )
    assert len(full) == 4
    assert len(resumed) == 2
    for index in range(len(resumed)):
        full_x, full_y, full_mask = full[index + 2]
        resumed_x, resumed_y, resumed_mask = resumed[index]
        assert torch.equal(resumed_x, full_x)
        assert torch.equal(resumed_y, full_y)
        assert torch.equal(resumed_mask, full_mask)
        assert torch.equal(resumed_mask, torch.tensor(records[index + 2].target_mask, dtype=torch.float32))
