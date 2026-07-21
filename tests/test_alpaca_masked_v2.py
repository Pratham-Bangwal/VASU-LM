import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.prepare_alpaca_masked_v2 import TOKEN_FILE
from vasu.data.alpaca_masked_v2 import (
    FORMAT_VERSION,
    build_metadata,
    build_packed_records,
    encode_source_example,
    validate_metadata_against_arrays,
)
from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.training.instruction_dataset import (
    InstructionDataset,
    PackedInstructionDataset,
)


PAD_ID = 0
EOS_ID = 3


class _CharacterTokenizer:
    def encode(self, text):
        return [ord(character) + 10 for character in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(token_id - 10) for token_id in ids)


TOKENIZER = _CharacterTokenizer()


def _example(instruction="Do this.", input_text="", output="Done."):
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def _records(examples, sequence_length=128):
    return build_packed_records(
        examples,
        TOKENIZER,
        sequence_length,
        PAD_ID,
        EOS_ID,
    )


def test_instruction_only_uses_exact_prompt():
    encoded = encode_source_example(_example(), TOKENIZER)
    assert TOKENIZER.decode(encoded.prompt_ids) == (
        "User: Do this.\nAssistant:"
    )


def test_optional_input_is_on_its_own_line():
    encoded = encode_source_example(
        _example(input_text="Context here."), TOKENIZER
    )
    assert TOKENIZER.decode(encoded.prompt_ids) == (
        "User: Do this.\nContext here.\nAssistant:"
    )


def test_empty_optional_input_is_omitted():
    assert format_alpaca_prompt("Do this.", " \n") == (
        "User: Do this.\nAssistant:"
    )


def test_assistant_mask_starts_after_complete_header():
    tokens, mask, _ = _records([_example()])
    first_supervised = int(np.flatnonzero(mask[0] == 1)[0])
    assert TOKENIZER.decode(tokens[0, :first_supervised].tolist()).endswith(
        "Assistant:"
    )
    assert TOKENIZER.decode(
        [int(tokens[0, first_supervised])]
    ) == " "


def test_eos_is_supervised_and_prompt_is_not():
    tokens, mask, stats = _records([_example()])
    eos_positions = tokens == EOS_ID
    assert stats.eos_tokens == 1
    assert np.all(mask[eos_positions] == 1)
    first_supervised = int(np.flatnonzero(mask[0] == 1)[0])
    assert np.all(mask[0, :first_supervised] == 0)


def test_padding_is_never_supervised():
    tokens, mask, _ = _records([_example()])
    assert np.all(mask[tokens == PAD_ID] == 0)


def test_shifted_target_mask_alignment(tmp_path):
    tokens, mask, _ = _records([_example()], sequence_length=64)
    token_file = tmp_path / "tokens.bin"
    mask_file = tmp_path / "mask.bin"
    tokens.reshape(-1).tofile(token_file)
    mask.reshape(-1).tofile(mask_file)
    dataset = PackedInstructionDataset(
        str(token_file), str(mask_file), seq_len=64
    )
    x, y, target_mask = dataset[0]
    assert x.tolist() == tokens[0, :-1].tolist()
    assert y.tolist() == tokens[0, 1:].tolist()
    assert target_mask.tolist() == mask[0, 1:].astype(float).tolist()


def test_multiple_examples_have_supervised_eos_boundaries():
    tokens, mask, stats = _records(
        [_example("A", output="B"), _example("C", output="D")],
        sequence_length=96,
    )
    eos_positions = np.flatnonzero(tokens[0] == EOS_ID)
    assert stats.eos_tokens == 2
    assert len(eos_positions) == 2
    assert np.all(mask[0, eos_positions] == 1)
    assert mask[0, eos_positions[0] + 1] == 0


def test_truncated_response_does_not_get_false_eos():
    tokens, mask, stats = _records(
        [_example("A", output="response " * 20)],
        sequence_length=32,
    )
    assert stats.truncated_examples == 1
    assert stats.eos_tokens == 0
    assert not np.any((tokens == EOS_ID) & (mask == 1))


def test_token_and_mask_lengths_are_equal():
    tokens, mask, _ = _records([_example()])
    assert tokens.shape == mask.shape
    assert tokens.dtype == np.uint16
    assert mask.dtype == np.uint8


def test_metadata_matches_actual_records():
    tokens, mask, stats = _records([_example()])
    metadata = build_metadata(
        tokens,
        mask,
        stats,
        tokenizer_path="tokenizer.json",
        tokenizer_vocab_size=32000,
        sequence_length=128,
        source_dataset_path="alpaca.jsonl",
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        seed=42,
    )
    validate_metadata_against_arrays(tokens, mask, metadata)
    assert metadata["format_version"] == FORMAT_VERSION
    assert metadata["number_of_source_examples"] == 1
    assert metadata["assistant_loss_tokens"] == int(mask.sum())


def test_empty_assistant_response_is_dropped_and_counted():
    tokens, mask, stats = _records(
        [_example(output=""), _example("valid", output="yes")]
    )
    assert tokens.shape == mask.shape
    assert stats.malformed_examples == 1
    assert stats.dropped_examples == 1


def test_malformed_source_example_is_counted():
    _, _, stats = _records(
        [{"instruction": 123, "input": "", "output": "bad"}, _example()]
    )
    assert stats.malformed_examples == 1
    assert stats.retained_examples == 1


def test_preparation_is_deterministic():
    examples = [_example("A", output="B"), _example("C", output="D")]
    first_tokens, first_mask, first_stats = _records(examples)
    second_tokens, second_mask, second_stats = _records(examples)
    assert np.array_equal(first_tokens, second_tokens)
    assert np.array_equal(first_mask, second_mask)
    assert first_stats == second_stats


def test_standard_alpaca_output_path_is_not_reused(tmp_path):
    standard = tmp_path / "alpaca.bin"
    standard.write_bytes(b"existing-standard-data")
    before = hashlib.sha256(standard.read_bytes()).hexdigest()
    assert TOKEN_FILE.name == "alpaca_masked_v2.bin"
    assert hashlib.sha256(standard.read_bytes()).hexdigest() == before


def test_existing_ultrachat_stream_dataset_remains_compatible(tmp_path):
    token_file = tmp_path / "ultrachat.bin"
    mask_file = tmp_path / "ultrachat_mask.bin"
    np.asarray([10, 11, 12, 13, 14], dtype=np.uint16).tofile(token_file)
    np.asarray([0, 0, 1, 1, 0], dtype=np.uint8).tofile(mask_file)
    dataset = InstructionDataset(
        str(token_file), str(mask_file), seq_len=4
    )
    _, y, target_mask = dataset[0]
    assert y.tolist() == [11, 12, 13, 14]
    assert target_mask.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_metadata_validation_rejects_length_mismatch():
    tokens, mask, stats = _records([_example()])
    metadata = build_metadata(
        tokens,
        mask,
        stats,
        tokenizer_path="tokenizer.json",
        tokenizer_vocab_size=32000,
        sequence_length=128,
        source_dataset_path="alpaca.jsonl",
        pad_token_id=PAD_ID,
        eos_token_id=EOS_ID,
        seed=42,
    )
    with pytest.raises(ValueError, match="shapes do not match"):
        validate_metadata_against_arrays(tokens, mask[:, :-1], metadata)
