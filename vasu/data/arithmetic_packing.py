"""Deterministic fixed-record packing for verified arithmetic examples.

The production contract is intentionally independent of tokenizer, dataset,
and trainer code. Examples are tokenized before they reach this module and
must already end with EOS. They are greedily packed, without splitting or
truncating, into fixed ``sequence_length + 1`` token records. Unused record
tails contain the caller-provided PAD token and are fully excluded from loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence
import warnings

import numpy as np


PACKED_ARITHMETIC_FORMAT = "vasu_packed_arithmetic_v2"
DEFAULT_SEQUENCE_LENGTH = 256


@dataclass(frozen=True)
class TokenizedArithmeticExample:
    """A complete, EOS-terminated arithmetic example ready for packing."""

    source_id: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class PackedExampleSpan:
    """One complete example's placement inside a packed record."""

    source_id: str
    start: int
    end: int
    replay_epoch: int


@dataclass(frozen=True)
class PackedArithmeticRecord:
    """One independent fixed-width record and explicit packing provenance.

    ``target_mask`` is the authoritative next-token mask and has exactly
    ``sequence_length`` positions. ``stored_mask`` is a lossless compatibility
    view for the existing :class:`PackedInstructionDataset`, which expects a
    record-length mask and exposes ``stored_mask[1:]`` to the loss function.
    """

    tokens: np.ndarray
    target_mask: np.ndarray
    spans: tuple[PackedExampleSpan, ...]
    example_ids: tuple[str, ...]
    replay_epoch: int
    used_token_count: int
    padding_token_count: int
    utilization_ratio: float

    @property
    def stored_mask(self) -> np.ndarray:
        """Return the legacy record-length representation of ``target_mask``."""

        mask = np.zeros(len(self.tokens), dtype=np.uint8)
        mask[1:] = self.target_mask
        return mask

    @property
    def token_mask(self) -> np.ndarray:
        """Backward-compatible alias for the legacy record-length mask."""

        return self.stored_mask


@dataclass(frozen=True)
class PackingStatistics:
    """Aggregate, reproducible accounting for a packed record collection."""

    packed_records: int
    total_logical_examples_consumed: int
    unique_examples_consumed: int
    replay_epochs: int
    real_tokens: int
    padding_tokens: int
    mean_utilization: float
    minimum_utilization: float
    maximum_utilization: float
    padding_percentage: float


def _permutation(
    examples: Sequence[TokenizedArithmeticExample],
    *,
    seed: int,
    replay_epoch: int,
) -> list[int]:
    """Return a stable permutation for one complete source replay epoch."""

    def key(index: int) -> bytes:
        value = f"{seed}:{replay_epoch}:{examples[index].source_id}".encode(
            "utf-8"
        )
        return hashlib.sha256(value).digest()

    return sorted(range(len(examples)), key=key)


def _validate_examples(
    examples: Sequence[TokenizedArithmeticExample],
    *,
    eos_token_id: int,
    pad_token_id: int,
    record_length: int,
) -> None:
    if not examples:
        raise ValueError("at least one complete arithmetic example is required")
    if eos_token_id == pad_token_id:
        raise ValueError("EOS and PAD token IDs must be distinct")
    if not 0 <= eos_token_id <= np.iinfo(np.uint16).max:
        raise ValueError("EOS token ID is outside uint16 range")
    if not 0 <= pad_token_id <= np.iinfo(np.uint16).max:
        raise ValueError("PAD token ID is outside uint16 range")

    seen_ids: set[str] = set()
    for example in examples:
        if not example.source_id:
            raise ValueError("arithmetic examples require a non-empty source_id")
        if example.source_id in seen_ids:
            raise ValueError(f"duplicate arithmetic source_id: {example.source_id!r}")
        seen_ids.add(example.source_id)
        if len(example.token_ids) < 2:
            raise ValueError(
                f"example {example.source_id!r} has no supervised response token"
            )
        if len(example.token_ids) > record_length:
            raise ValueError(
                f"example {example.source_id!r} exceeds fixed record length"
            )
        if example.token_ids[-1] != eos_token_id:
            raise ValueError(
                f"example {example.source_id!r} is not EOS-terminated"
            )
        if any(
            token < 0 or token > np.iinfo(np.uint16).max
            for token in example.token_ids
        ):
            raise ValueError(f"example {example.source_id!r} has an invalid token ID")
        if pad_token_id in example.token_ids:
            raise ValueError(
                f"example {example.source_id!r} contains the reserved PAD token"
            )


def summarize_packed_records(
    records: Sequence[PackedArithmeticRecord],
) -> PackingStatistics:
    """Return aggregate usage and explicit padding accounting."""

    if not records:
        raise ValueError("at least one packed record is required for statistics")
    real_tokens = sum(record.used_token_count for record in records)
    padding_tokens = sum(record.padding_token_count for record in records)
    total_tokens = real_tokens + padding_tokens
    utilization = [record.utilization_ratio for record in records]
    replay_epochs = {record.replay_epoch for record in records}
    example_ids = {
        source_id for record in records for source_id in record.example_ids
    }
    return PackingStatistics(
        packed_records=len(records),
        total_logical_examples_consumed=sum(len(record.spans) for record in records),
        unique_examples_consumed=len(example_ids),
        replay_epochs=len(replay_epochs),
        real_tokens=real_tokens,
        padding_tokens=padding_tokens,
        mean_utilization=real_tokens / total_tokens,
        minimum_utilization=min(utilization),
        maximum_utilization=max(utilization),
        padding_percentage=padding_tokens / total_tokens,
    )


def pack_arithmetic_examples(
    examples: Sequence[TokenizedArithmeticExample],
    *,
    record_count: int,
    eos_token_id: int,
    pad_token_id: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    seed: int = 0,
    utilization_warning_threshold: float | None = None,
    strict_utilization: bool = False,
) -> list[PackedArithmeticRecord]:
    """Greedily pack complete examples and explicitly PAD unused record tails.

    Each replay epoch has one deterministic source order. The next source
    example is placed if it fits; otherwise the current record is finalized
    with PAD and that same example begins the next record. A source replay does
    not begin until the prior epoch's final example has been emitted. Thus no
    example is skipped merely to improve utilization.

    The returned ``target_mask`` has ``sequence_length`` positions and is
    aligned with ``y = tokens[1:]``. Every real within-example target, including
    EOS, is enabled. EOS-to-next-example and every PAD-involved transition are
    disabled.
    """

    if record_count < 1:
        raise ValueError("record_count must be at least one")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if utilization_warning_threshold is not None and not 0 <= utilization_warning_threshold <= 1:
        raise ValueError("utilization_warning_threshold must be in [0, 1]")

    record_length = sequence_length + 1
    _validate_examples(
        examples,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        record_length=record_length,
    )

    records: list[PackedArithmeticRecord] = []
    replay_epoch = 0
    order = _permutation(examples, seed=seed, replay_epoch=replay_epoch)
    order_position = 0

    for _ in range(record_count):
        tokens = np.full(record_length, pad_token_id, dtype=np.uint16)
        target_mask = np.zeros(sequence_length, dtype=np.uint8)
        spans: list[PackedExampleSpan] = []
        offset = 0
        record_epoch = replay_epoch

        while True:
            if order_position == len(order):
                if offset:
                    break
                replay_epoch += 1
                order = _permutation(
                    examples,
                    seed=seed,
                    replay_epoch=replay_epoch,
                )
                order_position = 0
                record_epoch = replay_epoch

            example = examples[order[order_position]]
            example_length = len(example.token_ids)
            if offset + example_length > record_length:
                break

            end = offset + example_length
            tokens[offset:end] = np.asarray(example.token_ids, dtype=np.uint16)
            # target_mask[j] corresponds to y[j] == tokens[j + 1].  This range
            # includes EOS but excludes the first token of every example.
            target_mask[offset : end - 1] = 1
            spans.append(
                PackedExampleSpan(
                    source_id=example.source_id,
                    start=offset,
                    end=end,
                    replay_epoch=replay_epoch,
                )
            )
            offset = end
            order_position += 1
            if offset == record_length:
                break

        if not spans:
            raise RuntimeError("packing produced a PAD-only record")
        padding_token_count = record_length - offset
        record = PackedArithmeticRecord(
            tokens=tokens,
            target_mask=target_mask,
            spans=tuple(spans),
            example_ids=tuple(span.source_id for span in spans),
            replay_epoch=record_epoch,
            used_token_count=offset,
            padding_token_count=padding_token_count,
            utilization_ratio=offset / record_length,
        )
        records.append(record)

    statistics = summarize_packed_records(records)
    if (
        utilization_warning_threshold is not None
        and statistics.mean_utilization < utilization_warning_threshold
    ):
        message = (
            "arithmetic packing utilization is below the configured threshold: "
            f"{statistics.mean_utilization:.4f} < {utilization_warning_threshold:.4f}"
        )
        if strict_utilization:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return records
