from __future__ import annotations

import pytest

from vasu.data.arithmetic_step_supervision import (
    compile_supervised_example,
    pack_supervised_examples,
)
from vasu.data.arithmetic_v2 import generate_records


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) + 10 for character in text]


def _records() -> list[dict]:
    return generate_records("train", 220, 42)[:8]


def test_matched_variants_have_the_same_unsupervised_prompt() -> None:
    record = _records()[0]
    tokenizer = CharacterTokenizer()
    control = compile_supervised_example(
        record, tokenizer=tokenizer, eos_token_id=3, variant="final_answer"
    )
    treatment = compile_supervised_example(
        record, tokenizer=tokenizer, eos_token_id=3, variant="verified_steps"
    )

    assert control.token_ids[: control.target_start] == treatment.token_ids[: treatment.target_start]
    assert set(control.stored_mask[: control.target_start]) == {0}
    assert set(treatment.stored_mask[: treatment.target_start]) == {0}
    assert all(control.stored_mask[control.target_start :])
    assert all(treatment.stored_mask[treatment.target_start :])
    assert control.token_ids[-1] == treatment.token_ids[-1] == 3
    assert sum(treatment.stored_mask) > sum(control.stored_mask)


def test_packing_preserves_masks_and_disables_padding() -> None:
    tokenizer = CharacterTokenizer()
    examples = [
        compile_supervised_example(
            record, tokenizer=tokenizer, eos_token_id=3, variant="verified_steps"
        )
        for record in _records()
    ]
    packed = pack_supervised_examples(examples, pad_token_id=0, sequence_length=256)

    assert sum(len(record.example_ids) for record in packed) == len(examples)
    for record in packed:
        assert record.tokens.shape == record.stored_mask.shape == (257,)
        assert not record.stored_mask[record.used_token_count :].any()
        assert not (record.tokens[record.used_token_count :] != 0).any()


def test_compiler_rejects_merged_prompt_response_boundary() -> None:
    class MergingTokenizer(CharacterTokenizer):
        def encode(self, text: str) -> list[int]:
            ids = super().encode(text)
            return ids if text.endswith("\n") else [999, *ids[1:]]

    with pytest.raises(ValueError, match="merged"):
        compile_supervised_example(
            _records()[0],
            tokenizer=MergingTokenizer(),
            eos_token_id=3,
            variant="final_answer",
        )
