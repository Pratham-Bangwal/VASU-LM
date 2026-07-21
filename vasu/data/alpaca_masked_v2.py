"""Boundary-preserving Alpaca packing for assistant-only supervision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.tokenizer.tokenizer import VASUTokenizer


FORMAT_VERSION = "alpaca_masked_v2_record_packed_eos_v1"
PACKING_STRATEGY = (
    "fixed 257-token records; complete examples packed with supervised EOS; "
    "overlong responses truncated without EOS; PAD masked"
)
SPECIAL_TOKEN_TEXT = ("[PAD]", "[UNK]", "[BOS]", "[EOS]")


class MalformedExampleError(ValueError):
    """Raised when an Alpaca source row cannot be safely serialized."""


@dataclass(frozen=True)
class EncodedExample:
    prompt_ids: list[int]
    response_ids: list[int]


@dataclass
class PackingStats:
    number_of_source_examples: int = 0
    retained_examples: int = 0
    truncated_examples: int = 0
    dropped_examples: int = 0
    malformed_examples: int = 0
    prompt_tokens: int = 0
    assistant_loss_tokens: int = 0
    eos_tokens: int = 0


def encode_source_example(
    sample: dict[str, Any],
    tokenizer: VASUTokenizer,
) -> EncodedExample:
    """Encode the exact shared Alpaca prompt and its response separately."""
    if not isinstance(sample, dict):
        raise MalformedExampleError("source example must be a JSON object")

    instruction = sample.get("instruction")
    response = sample.get("output")
    input_text = sample.get("input", "")

    if not isinstance(instruction, str) or not instruction.strip():
        raise MalformedExampleError("instruction must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise MalformedExampleError("output must be a non-empty string")
    if input_text is None:
        input_text = ""
    if not isinstance(input_text, str):
        raise MalformedExampleError("input must be a string")

    fields = (instruction, input_text, response)
    if any(marker in field for marker in SPECIAL_TOKEN_TEXT for field in fields):
        raise MalformedExampleError(
            "source text contains a reserved tokenizer marker"
        )

    prompt = format_alpaca_prompt(
        instruction=instruction,
        input_text=input_text,
        include_response_header=True,
    )
    prompt_ids = tokenizer.encode(prompt)
    # The existing standard format places exactly one space after Assistant:.
    response_ids = tokenizer.encode(f" {response}")
    if not prompt_ids or not response_ids:
        raise MalformedExampleError("tokenization produced an empty segment")

    return EncodedExample(prompt_ids=prompt_ids, response_ids=response_ids)


def build_packed_records(
    examples: Iterable[dict[str, Any]],
    tokenizer: VASUTokenizer,
    sequence_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> tuple[np.ndarray, np.ndarray, PackingStats]:
    """Pack examples without allowing the dataset loader to cross records."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")

    record_length = sequence_length + 1
    records: list[np.ndarray] = []
    mask_records: list[np.ndarray] = []
    current_tokens: list[int] = []
    current_mask: list[int] = []
    stats = PackingStats()

    def flush_record() -> None:
        nonlocal current_tokens, current_mask
        if not current_tokens:
            return
        padding = record_length - len(current_tokens)
        if padding < 0:
            raise AssertionError("record exceeded its fixed length")
        current_tokens.extend([pad_token_id] * padding)
        current_mask.extend([0] * padding)
        record = np.asarray(current_tokens, dtype=np.uint16)
        mask_record = np.asarray(current_mask, dtype=np.uint8)
        if not mask_record[1:].any():
            raise ValueError("packed record has zero supervised target tokens")
        records.append(record)
        mask_records.append(mask_record)
        current_tokens = []
        current_mask = []

    for sample in examples:
        stats.number_of_source_examples += 1
        try:
            encoded = encode_source_example(sample, tokenizer)
        except MalformedExampleError:
            stats.malformed_examples += 1
            stats.dropped_examples += 1
            continue

        complete_tokens = [
            *encoded.prompt_ids,
            *encoded.response_ids,
            eos_token_id,
        ]
        complete_mask = [
            *([0] * len(encoded.prompt_ids)),
            *([1] * (len(encoded.response_ids) + 1)),
        ]

        if len(complete_tokens) <= record_length:
            if current_tokens and (
                len(current_tokens) + len(complete_tokens) > record_length
            ):
                flush_record()
            current_tokens.extend(complete_tokens)
            current_mask.extend(complete_mask)
            stats.retained_examples += 1
            stats.prompt_tokens += len(encoded.prompt_ids)
            stats.assistant_loss_tokens += len(encoded.response_ids) + 1
            stats.eos_tokens += 1
            continue

        # An overlong example owns a fresh record so its partial response cannot
        # be followed by an unrelated example without a real EOS boundary.
        flush_record()
        response_capacity = record_length - len(encoded.prompt_ids)
        if response_capacity <= 0:
            stats.dropped_examples += 1
            continue

        truncated_response = encoded.response_ids[:response_capacity]
        if not truncated_response:
            stats.dropped_examples += 1
            continue

        current_tokens.extend(encoded.prompt_ids)
        current_tokens.extend(truncated_response)
        current_mask.extend([0] * len(encoded.prompt_ids))
        current_mask.extend([1] * len(truncated_response))
        stats.retained_examples += 1
        stats.truncated_examples += 1
        stats.prompt_tokens += len(encoded.prompt_ids)
        stats.assistant_loss_tokens += len(truncated_response)
        # Deliberately no EOS: the source response end is not present.
        flush_record()

    flush_record()

    if not records:
        raise ValueError("no valid supervised training records were produced")

    token_array = np.stack(records)
    mask_array = np.stack(mask_records)
    if token_array.shape != mask_array.shape:
        raise AssertionError("token and mask record shapes differ")
    return token_array, mask_array, stats


def build_metadata(
    token_records: np.ndarray,
    mask_records: np.ndarray,
    stats: PackingStats,
    *,
    tokenizer_path: str,
    tokenizer_vocab_size: int,
    sequence_length: int,
    source_dataset_path: str,
    pad_token_id: int,
    eos_token_id: int,
    seed: int,
) -> dict[str, Any]:
    """Create auditable metadata from the actual packed arrays."""
    total_tokens = int(token_records.size)
    padding_tokens = int((token_records == pad_token_id).sum())
    non_padding_tokens = total_tokens - padding_tokens
    ratio = (
        stats.assistant_loss_tokens / non_padding_tokens
        if non_padding_tokens
        else 0.0
    )
    return {
        "format_version": FORMAT_VERSION,
        "tokenizer_path": tokenizer_path,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "sequence_length": sequence_length,
        "record_length": sequence_length + 1,
        "number_of_source_examples": stats.number_of_source_examples,
        "number_of_training_records": int(token_records.shape[0]),
        "retained_examples": stats.retained_examples,
        "total_tokens": total_tokens,
        "assistant_loss_tokens": stats.assistant_loss_tokens,
        "prompt_tokens": stats.prompt_tokens,
        "padding_tokens": padding_tokens,
        "eos_tokens": stats.eos_tokens,
        "assistant_token_ratio": ratio,
        "truncated_examples": stats.truncated_examples,
        "dropped_examples": stats.dropped_examples,
        "malformed_examples": stats.malformed_examples,
        "packing_strategy": PACKING_STRATEGY,
        "source_dataset_path": source_dataset_path,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "seed": seed,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def atomic_write_array(path: Path, array: np.ndarray) -> None:
    """Write a binary array completely before atomically publishing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        array.tofile(temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically publish UTF-8 JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def validate_metadata_against_arrays(
    token_records: np.ndarray,
    mask_records: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Validate structural invariants and recorded aggregate statistics."""
    if token_records.shape != mask_records.shape:
        raise ValueError("token and mask shapes do not match")
    if token_records.ndim != 2:
        raise ValueError("packed token data must be two-dimensional")
    if not np.isin(mask_records, (0, 1)).all():
        raise ValueError("mask contains values other than 0 or 1")

    pad_id = int(metadata["pad_token_id"])
    eos_id = int(metadata["eos_token_id"])
    if np.any(mask_records[token_records == pad_id] != 0):
        raise ValueError("padding token has a non-zero loss mask")
    if np.any(mask_records[:, 1:].sum(axis=1) == 0):
        raise ValueError("record with zero supervised target tokens found")

    actual = {
        "number_of_training_records": int(token_records.shape[0]),
        "total_tokens": int(token_records.size),
        "assistant_loss_tokens": int(mask_records.sum()),
        "prompt_tokens": int(
            ((mask_records == 0) & (token_records != pad_id)).sum()
        ),
        "padding_tokens": int((token_records == pad_id).sum()),
        "eos_tokens": int(
            ((token_records == eos_id) & (mask_records == 1)).sum()
        ),
    }
    for key, value in actual.items():
        if int(metadata[key]) != value:
            raise ValueError(
                f"metadata mismatch for {key}: {metadata[key]} != {value}"
            )

    if int(metadata["record_length"]) != token_records.shape[1]:
        raise ValueError("metadata record_length does not match arrays")
    if int(metadata["sequence_length"]) + 1 != token_records.shape[1]:
        raise ValueError("metadata sequence_length does not match arrays")
