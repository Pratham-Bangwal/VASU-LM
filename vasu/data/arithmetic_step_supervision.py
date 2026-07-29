"""Loss-masked paired targets for the proposed Candidate E arithmetic study.

This module intentionally only compiles logical, metadata-verified arithmetic
records into model-ready examples.  It does not write a dataset release or
configure or start training.  The paired variants share an identical prompt;
only the supervised response serialization differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from vasu.data.arithmetic_steps import serialize_verified_steps
from vasu.data.arithmetic_v2 import normalized_expression_sha256, recompute_answer


TargetVariant = Literal["final_answer", "verified_steps"]
PROMPT_PREFIX_FORMAT = "Question: {prompt}\nResponse:\n"


class TextTokenizer(Protocol):
    """Minimal tokenizer interface needed to compile supervised examples."""

    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class SupervisedArithmeticExample:
    """An EOS-terminated example with an explicit token-level loss mask."""

    source_id: str
    token_ids: tuple[int, ...]
    stored_mask: tuple[int, ...]
    target_start: int


@dataclass(frozen=True)
class PackedSupervisedArithmeticRecord:
    """A fixed-width packed record preserving each example's loss boundary."""

    tokens: np.ndarray
    stored_mask: np.ndarray
    example_ids: tuple[str, ...]
    used_token_count: int


def validate_matched_logical_splits(
    control_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    treatment_splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    """Prove paired-source equality and semantic isolation before release work.

    A future release constructor must call this before serializing either arm.
    It rejects duplicate IDs, prompts, or verified semantic expressions across
    splits and requires each treatment record to be the same logical example as
    its control counterpart.  This check is deliberately independent from the
    target serialization so it cannot be satisfied by merely matching text.
    """

    expected_splits = {"train", "development", "evaluation"}
    if set(control_splits) != expected_splits or set(treatment_splits) != expected_splits:
        raise ValueError("control and treatment require train, development, and evaluation splits")

    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    seen_expressions: set[str] = set()
    counts: dict[str, int] = {}
    for split in sorted(expected_splits):
        control_by_id = {str(record["id"]): record for record in control_splits[split]}
        treatment_by_id = {str(record["id"]): record for record in treatment_splits[split]}
        if not control_by_id or len(control_by_id) != len(control_splits[split]):
            raise ValueError(f"control {split} contains missing or duplicate IDs")
        if set(control_by_id) != set(treatment_by_id) or len(treatment_by_id) != len(treatment_splits[split]):
            raise ValueError(f"control and treatment {split} source IDs differ")
        for source_id, control in control_by_id.items():
            treatment = treatment_by_id[source_id]
            if (
                control["prompt"] != treatment["prompt"]
                or recompute_answer(control) != recompute_answer(treatment)
                or normalized_expression_sha256(control)
                != normalized_expression_sha256(treatment)
            ):
                raise ValueError(f"paired source mismatch for {source_id!r}")
            prompt = str(control["prompt"])
            expression = normalized_expression_sha256(control)
            if source_id in seen_ids or prompt in seen_prompts or expression in seen_expressions:
                raise ValueError(f"split isolation violation at {source_id!r}")
            seen_ids.add(source_id)
            seen_prompts.add(prompt)
            seen_expressions.add(expression)
        counts[split] = len(control_by_id)
    return counts


def response_text(record: Mapping[str, Any], variant: TargetVariant) -> str:
    """Return the sole experimental difference between the matched variants."""

    answer = recompute_answer(record)
    if variant == "final_answer":
        return f"Answer: {answer}"
    if variant == "verified_steps":
        return serialize_verified_steps(record)
    raise ValueError(f"unsupported target variant: {variant!r}")


def compile_supervised_example(
    record: Mapping[str, Any],
    *,
    tokenizer: TextTokenizer,
    eos_token_id: int,
    variant: TargetVariant,
) -> SupervisedArithmeticExample:
    """Compile one record while fail-closing on a tokenizer-boundary mismatch.

    ``stored_mask[i]`` supervises ``token_ids[i]``.  Prompt tokens are always
    zero.  Every response token and EOS is one, so the caller can derive the
    trainer's next-token mask as ``stored_mask[1:]`` without supervising
    cross-example boundaries.
    """

    source_id = str(record["id"])
    if not source_id:
        raise ValueError("arithmetic record requires a non-empty id")
    prefix = PROMPT_PREFIX_FORMAT.format(prompt=record["prompt"])
    full_text = prefix + response_text(record, variant)
    prefix_ids = tuple(tokenizer.encode(prefix))
    full_ids = tuple(tokenizer.encode(full_text))
    if not prefix_ids or len(full_ids) <= len(prefix_ids):
        raise ValueError(f"record {source_id!r} has no tokenized response")
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "tokenizer merged the prompt/response boundary; choose a stable "
            "serialization boundary before preparing a release"
        )
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise ValueError("eos_token_id must be a non-negative integer")

    token_ids = (*full_ids, eos_token_id)
    target_start = len(prefix_ids)
    stored_mask = (0,) * target_start + (1,) * (len(token_ids) - target_start)
    return SupervisedArithmeticExample(
        source_id=source_id,
        token_ids=token_ids,
        stored_mask=stored_mask,
        target_start=target_start,
    )


def pack_supervised_examples(
    examples: Sequence[SupervisedArithmeticExample],
    *,
    pad_token_id: int,
    sequence_length: int = 256,
) -> list[PackedSupervisedArithmeticRecord]:
    """Pack complete examples without changing their prompt/target masks."""

    if not examples:
        raise ValueError("at least one example is required")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    record_length = sequence_length + 1
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")

    records: list[PackedSupervisedArithmeticRecord] = []
    tokens = np.full(record_length, pad_token_id, dtype=np.uint16)
    mask = np.zeros(record_length, dtype=np.uint8)
    source_ids: list[str] = []
    offset = 0

    def finalize() -> None:
        nonlocal tokens, mask, source_ids, offset
        if source_ids:
            records.append(
                PackedSupervisedArithmeticRecord(
                    tokens=tokens,
                    stored_mask=mask,
                    example_ids=tuple(source_ids),
                    used_token_count=offset,
                )
            )
        tokens = np.full(record_length, pad_token_id, dtype=np.uint16)
        mask = np.zeros(record_length, dtype=np.uint8)
        source_ids = []
        offset = 0

    seen_ids: set[str] = set()
    for example in examples:
        if not example.source_id or example.source_id in seen_ids:
            raise ValueError("examples require unique, non-empty source IDs")
        seen_ids.add(example.source_id)
        if len(example.token_ids) != len(example.stored_mask):
            raise ValueError(f"mask length mismatch for {example.source_id!r}")
        if not 0 < example.target_start < len(example.token_ids):
            raise ValueError(f"invalid target boundary for {example.source_id!r}")
        if any(example.stored_mask[: example.target_start]):
            raise ValueError(f"prompt tokens are supervised for {example.source_id!r}")
        if not all(example.stored_mask[example.target_start :]):
            raise ValueError(f"response tokens are not fully supervised for {example.source_id!r}")
        if len(example.token_ids) > record_length:
            raise ValueError(f"example {example.source_id!r} exceeds fixed record length")
        if pad_token_id in example.token_ids:
            raise ValueError(f"example {example.source_id!r} contains PAD")
        if offset and offset + len(example.token_ids) > record_length:
            finalize()

        end = offset + len(example.token_ids)
        tokens[offset:end] = np.asarray(example.token_ids, dtype=np.uint16)
        mask[offset:end] = np.asarray(example.stored_mask, dtype=np.uint8)
        source_ids.append(example.source_id)
        offset = end
        if offset == record_length:
            finalize()
    finalize()
    return records
