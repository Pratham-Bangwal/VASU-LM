from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_ultrachat_masked_v2 import TOKEN_FILE
from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.data.ultrachat_masked_v2 import (
    EncodedTurn,
    FORMAT_VERSION,
    TurnRecordPacker,
    build_metadata,
    encode_conversation,
    validate_records,
)
from vasu.training.instruction_dataset import (
    InstructionDataset,
    PackedInstructionDataset,
)
import train_vasu_60m_ultrachat_masked_v2_from_alpaca_v3 as runner


PAD_ID = 0
EOS_ID = 3


class CharacterTokenizer:
    def __init__(self):
        self.encoded_texts: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.encoded_texts.append(text)
        return [ord(character) + 10 for character in text]

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(chr(value - 10) for value in ids)


def conversation(*pairs: tuple[str, str]) -> dict:
    messages = []
    for user, assistant in pairs:
        messages.extend(
            (
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            )
        )
    return {"prompt_id": "synthetic", "messages": messages}


def packer(sequence_length: int = 128) -> TurnRecordPacker:
    return TurnRecordPacker(
        sequence_length=sequence_length,
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
    )


def metadata(tokens, mask, stats, sequence_length=128):
    return build_metadata(
        tokens=tokens,
        mask=mask,
        stats=stats,
        source_path=Path("ultrachat.jsonl"),
        source_sha256="a" * 64,
        source_examples_inspected=stats.examples_retained,
        selection_stop_example_excluded=1,
        tokenizer_path=Path("assets/tokenizer.json"),
        tokenizer_sha256="b" * 64,
        vocabulary_size=65_536,
        pad_token_id=PAD_ID,
        bos_token_id=2,
        eos_token_id=EOS_ID,
        unk_token_id=1,
        sequence_length=sequence_length,
        split_seed=42,
        token_file_sha256="c" * 64,
        mask_file_sha256="d" * 64,
    )


def test_canonical_turn_serialization_uses_shared_prompt():
    tokenizer = CharacterTokenizer()
    turns = encode_conversation(
        conversation(("Explain gravity", "It pulls objects.")),
        tokenizer,
        forbidden_token_ids=frozenset((0, 2, 3)),
    )
    assert tokenizer.decode(list(turns[0].prompt_ids)) == format_alpaca_prompt(
        "Explain gravity"
    )
    assert tokenizer.decode(list(turns[0].response_ids)) == " It pulls objects."


def test_multiturn_source_order_is_preserved():
    tokenizer = CharacterTokenizer()
    turns = encode_conversation(
        conversation(("First", "One"), ("Second", "Two")),
        tokenizer,
        forbidden_token_ids=frozenset((0, 2, 3)),
    )
    assert [tokenizer.decode(list(turn.response_ids)) for turn in turns] == [
        " One",
        " Two",
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "only"}],
        [
            {"role": "assistant", "content": "wrong"},
            {"role": "user", "content": "order"},
        ],
    ],
)
def test_malformed_conversation_is_rejected(messages):
    with pytest.raises(ValueError):
        encode_conversation(
            {"messages": messages},
            CharacterTokenizer(),
            forbidden_token_ids=frozenset((0, 2, 3)),
        )


def test_complete_response_and_eos_are_supervised():
    value = packer(32)
    value.add_conversation([EncodedTurn((10, 11), (20, 21))])
    tokens, mask = value.finalize()
    assert tokens[0, :5].tolist() == [10, 11, 20, 21, EOS_ID]
    assert mask[0, :5].tolist() == [0, 0, 1, 1, 1]


def test_complete_turns_pack_only_across_supervised_eos():
    value = packer(16)
    value.add_conversation([EncodedTurn((10,), (20,))])
    value.add_conversation([EncodedTurn((11,), (21,))])
    tokens, mask = value.finalize()
    eos_positions = np.flatnonzero(tokens[0] == EOS_ID)
    assert eos_positions.tolist() == [2, 5]
    assert np.all(mask[0, eos_positions] == 1)


def test_overlong_response_is_truncated_without_false_eos():
    value = packer(7)
    value.add_conversation([EncodedTurn((10, 11), tuple(range(20, 40)))])
    tokens, mask = value.finalize()
    assert value.stats.turns_truncated == 1
    assert not np.any((tokens == EOS_ID) & (mask == 1))
    assert mask[0].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]


def test_no_later_turn_follows_a_truncated_turn():
    value = packer(7)
    value.add_conversation(
        [
            EncodedTurn((10, 11), tuple(range(20, 40))),
            EncodedTurn((12,), (50,)),
        ]
    )
    tokens, _ = value.finalize()
    assert 50 not in tokens
    assert value.stats.turns_dropped == 1


def test_prompt_too_long_drops_remaining_conversation():
    value = packer(7)
    value.add_conversation(
        [EncodedTurn(tuple(range(10, 19)), (30,)), EncodedTurn((12,), (50,))]
    )
    assert value.stats.examples_dropped == 1
    assert value.stats.turns_dropped == 2
    with pytest.raises(ValueError, match="no records"):
        value.finalize()


def test_padding_is_masked_and_shift_alignment_is_exact(tmp_path):
    value = packer(8)
    value.add_conversation([EncodedTurn((10, 11), (20,))])
    tokens, mask = value.finalize()
    assert np.all(mask[tokens == PAD_ID] == 0)
    token_path = tmp_path / "tokens.bin"
    mask_path = tmp_path / "mask.bin"
    tokens.tofile(token_path)
    mask.tofile(mask_path)
    dataset = PackedInstructionDataset(
        str(token_path), str(mask_path), seq_len=8
    )
    x, y, loss_mask = dataset[0]
    assert x.tolist() == tokens[0, :-1].tolist()
    assert y.tolist() == tokens[0, 1:].tolist()
    assert loss_mask.tolist() == mask[0, 1:].astype(float).tolist()


def test_metadata_and_record_validation_match_actual_arrays():
    value = packer(8)
    for index in range(20):
        value.add_conversation([EncodedTurn((10 + index,), (100 + index,))])
    tokens, mask = value.finalize()
    data = metadata(tokens, mask, value.stats, sequence_length=8)
    validate_records(tokens, mask, data)
    assert data["format_version"] == FORMAT_VERSION
    assert data["train_records"] + data["validation_records"] == len(tokens)
    assert data["stats"] == asdict(value.stats)


def test_invalid_mask_and_supervised_padding_are_rejected():
    value = packer(128)
    for index in range(20):
        value.add_conversation([EncodedTurn((10 + index,), (100 + index,))])
    tokens, mask = value.finalize()
    data = metadata(tokens, mask, value.stats)
    bad = mask.copy()
    bad[0, -1] = 2
    with pytest.raises(ValueError, match="mask contains"):
        validate_records(tokens, bad, data)
    bad = mask.copy()
    bad[0, -1] = 1
    with pytest.raises(ValueError, match="padding token"):
        validate_records(tokens, bad, data)


def test_snapshot_restore_makes_limit_selection_conversation_atomic():
    value = packer(7)
    value.add_conversation([EncodedTurn((10,), (20,))])
    snapshot = value.snapshot()
    value.add_conversation([EncodedTurn((11,), tuple(range(30, 50)))])
    value.restore(snapshot)
    tokens, _ = value.finalize()
    assert 11 not in tokens
    assert value.stats.examples_retained == 1


def test_preparation_is_deterministic():
    turns = [EncodedTurn((10,), (20,)), EncodedTurn((11,), (21,))]
    first = packer(16)
    second = packer(16)
    first.add_conversation(turns)
    second.add_conversation(turns)
    first_tokens, first_mask = first.finalize()
    second_tokens, second_mask = second.finalize()
    assert np.array_equal(first_tokens, second_tokens)
    assert np.array_equal(first_mask, second_mask)
    assert first.stats == second.stats


def test_legacy_ultrachat_dataset_and_output_path_are_untouched(tmp_path):
    legacy_tokens = tmp_path / "ultrachat.bin"
    legacy_mask = tmp_path / "ultrachat_mask.bin"
    np.asarray([10, 11, 12, 13, 14], dtype=np.uint16).tofile(legacy_tokens)
    np.asarray([0, 0, 1, 1, 0], dtype=np.uint8).tofile(legacy_mask)
    before = hashlib.sha256(legacy_tokens.read_bytes()).hexdigest()
    dataset = InstructionDataset(
        str(legacy_tokens), str(legacy_mask), seq_len=4
    )
    assert len(dataset) == 1
    assert TOKEN_FILE.name == "ultrachat_masked_v2.bin"
    assert hashlib.sha256(legacy_tokens.read_bytes()).hexdigest() == before


def test_training_runner_uses_isolated_alpaca_v3_start():
    assert runner.BASE_CHECKPOINT.as_posix().endswith(
        "alpaca_masked_v3_from_200k/best.pt"
    )
    assert runner.BASE_GLOBAL_STEP == 200_711
    assert runner.CHECKPOINT_DIR.as_posix().endswith(
        "ultrachat_masked_v2_from_alpaca_v3"
    )
    assert runner.BASE_CHECKPOINT.parent != runner.CHECKPOINT_DIR


def test_training_configuration_matches_controlled_contract():
    config = runner.build_train_config()
    assert config.epochs == 1
    assert config.batch_size == 2
    assert config.gradient_accumulation_steps == 16
    assert config.learning_rate == 2e-6
    assert config.weight_decay == 0.01
    assert config.grad_clip == 1.0
    assert config.use_amp is True


def test_split_and_optimizer_target_are_exact():
    train, validation, optimizer_steps = runner.split_counts(19_844)
    assert (train, validation) == (18_851, 993)
    assert optimizer_steps == 590
    assert runner.BASE_GLOBAL_STEP + optimizer_steps == 201_301


def test_output_isolation_rejects_a_protected_path(monkeypatch):
    monkeypatch.setattr(
        runner,
        "CHECKPOINT_DIR",
        Path("checkpoints/vasu_60m/alpaca_masked_v3_from_200k"),
    )
    with pytest.raises(ValueError, match="overlaps protected"):
        runner.validate_output_isolation()
