"""Boundary-aware fixed-record preparation for masked UltraChat v2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vasu.data.prompt_templates import format_alpaca_prompt
from vasu.tokenizer.tokenizer import VASUTokenizer


FORMAT_VERSION = "ultrachat_masked_v2_turn_packed_eos_v1"


@dataclass(frozen=True)
class EncodedTurn:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]


@dataclass
class PackingStats:
    examples_retained: int = 0
    examples_dropped: int = 0
    examples_truncated: int = 0
    turns_retained: int = 0
    turns_dropped: int = 0
    turns_truncated: int = 0
    malformed_examples: int = 0
    prompt_tokens: int = 0
    assistant_supervised_tokens: int = 0
    supervised_eos_count: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_conversation(
    sample: dict[str, Any],
    tokenizer: VASUTokenizer,
    *,
    forbidden_token_ids: frozenset[int],
) -> list[EncodedTurn]:
    """Encode alternating user/assistant turns using the inference template."""

    messages = sample.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) % 2:
        raise ValueError("messages must be a non-empty user/assistant pair list")
    turns: list[EncodedTurn] = []
    for index in range(0, len(messages), 2):
        user = messages[index]
        assistant = messages[index + 1]
        if not isinstance(user, dict) or not isinstance(assistant, dict):
            raise ValueError("message entries must be objects")
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError("messages must alternate user then assistant")
        user_text = user.get("content")
        response_text = assistant.get("content")
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user content must be non-empty text")
        if not isinstance(response_text, str) or not response_text.strip():
            raise ValueError("assistant content must be non-empty text")
        prompt_ids = tokenizer.encode(format_alpaca_prompt(user_text))
        response_ids = tokenizer.encode(f" {response_text}")
        if not prompt_ids or not response_ids:
            raise ValueError("encoded prompt and response must be non-empty")
        if forbidden_token_ids.intersection(prompt_ids):
            raise ValueError("prompt contains a reserved boundary token")
        if forbidden_token_ids.intersection(response_ids):
            raise ValueError("response contains a reserved boundary token")
        turns.append(
            EncodedTurn(tuple(prompt_ids), tuple(response_ids))
        )
    return turns


class TurnRecordPacker:
    """Pack complete turns only across supervised EOS boundaries."""

    def __init__(
        self,
        *,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.sequence_length = sequence_length
        self.record_length = sequence_length + 1
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.records: list[np.ndarray] = []
        self.mask_records: list[np.ndarray] = []
        self.current_tokens: list[int] = []
        self.current_mask: list[int] = []
        self.stats = PackingStats()

    def snapshot(self) -> tuple[int, list[int], list[int], PackingStats]:
        return (
            len(self.records),
            self.current_tokens.copy(),
            self.current_mask.copy(),
            deepcopy(self.stats),
        )

    def restore(
        self,
        snapshot: tuple[int, list[int], list[int], PackingStats],
    ) -> None:
        record_count, tokens, mask, stats = snapshot
        del self.records[record_count:]
        del self.mask_records[record_count:]
        self.current_tokens = tokens
        self.current_mask = mask
        self.stats = stats

    def projected_record_count(self) -> int:
        return len(self.records) + bool(self.current_tokens)

    def _flush(self) -> None:
        if not self.current_tokens:
            return
        padding = self.record_length - len(self.current_tokens)
        if padding < 0:
            raise AssertionError("record exceeded fixed length")
        tokens = self.current_tokens + [self.pad_token_id] * padding
        mask = self.current_mask + [0] * padding
        self.records.append(np.asarray(tokens, dtype=np.uint16))
        self.mask_records.append(np.asarray(mask, dtype=np.uint8))
        self.current_tokens = []
        self.current_mask = []

    def add_turn(self, turn: EncodedTurn) -> str:
        complete_length = (
            len(turn.prompt_ids) + len(turn.response_ids) + 1
        )
        if complete_length <= self.record_length:
            if len(self.current_tokens) + complete_length > self.record_length:
                self._flush()
            self.current_tokens.extend(turn.prompt_ids)
            self.current_mask.extend([0] * len(turn.prompt_ids))
            self.current_tokens.extend(turn.response_ids)
            self.current_mask.extend([1] * len(turn.response_ids))
            self.current_tokens.append(self.eos_token_id)
            self.current_mask.append(1)
            self.stats.turns_retained += 1
            self.stats.prompt_tokens += len(turn.prompt_ids)
            self.stats.assistant_supervised_tokens += len(turn.response_ids) + 1
            self.stats.supervised_eos_count += 1
            return "complete"

        # An overlong turn starts in a fresh record. Preserve the full prompt
        # and as much response as possible, but never synthesize an EOS for a
        # response whose true ending is absent.
        self._flush()
        response_capacity = self.record_length - len(turn.prompt_ids)
        if response_capacity <= 0:
            self.stats.turns_dropped += 1
            return "dropped"
        truncated_response = turn.response_ids[:response_capacity]
        if not truncated_response:
            self.stats.turns_dropped += 1
            return "dropped"
        self.current_tokens.extend(turn.prompt_ids)
        self.current_mask.extend([0] * len(turn.prompt_ids))
        self.current_tokens.extend(truncated_response)
        self.current_mask.extend([1] * len(truncated_response))
        self.stats.turns_retained += 1
        self.stats.turns_truncated += 1
        self.stats.prompt_tokens += len(turn.prompt_ids)
        self.stats.assistant_supervised_tokens += len(truncated_response)
        self._flush()  # Nothing may follow a truncated response in this record.
        return "truncated"

    def add_conversation(self, turns: list[EncodedTurn]) -> None:
        retained = 0
        truncated = False
        for turn_index, turn in enumerate(turns):
            result = self.add_turn(turn)
            if result != "dropped":
                retained += 1
            if result == "truncated":
                truncated = True
                # The remaining turns depend on conversation context whose
                # assistant response was truncated. Do not serialize them as
                # if that missing context had been complete.
                self.stats.turns_dropped += len(turns) - turn_index - 1
                break
            if result == "dropped":
                # A missing user/assistant pair likewise breaks the logical
                # conversation boundary for every following turn.
                self.stats.turns_dropped += len(turns) - turn_index - 1
                break
        if retained:
            self.stats.examples_retained += 1
            if truncated:
                self.stats.examples_truncated += 1
        else:
            self.stats.examples_dropped += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        self._flush()
        if not self.records:
            raise ValueError("packing produced no records")
        return np.stack(self.records), np.stack(self.mask_records)


def validate_records(
    tokens: np.ndarray,
    mask: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if tokens.shape != mask.shape or tokens.ndim != 2:
        raise ValueError("token and mask records must have equal 2D shapes")
    if tokens.dtype != np.uint16 or mask.dtype != np.uint8:
        raise ValueError("token/mask dtype must be uint16/uint8")
    if tokens.shape[1] != int(metadata["record_length"]):
        raise ValueError("record length does not match metadata")
    if int(metadata["sequence_length"]) + 1 != tokens.shape[1]:
        raise ValueError("sequence length does not match records")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("mask contains values other than 0 or 1")
    vocab_size = int(metadata["vocabulary_size"])
    if int(tokens.max()) >= vocab_size:
        raise ValueError("token ID is outside tokenizer vocabulary")
    pad_id = int(metadata["pad_token_id"])
    eos_id = int(metadata["eos_token_id"])
    if np.any(mask[tokens == pad_id] != 0):
        raise ValueError("padding token is supervised")
    if np.any(mask[:, 1:].sum(axis=1) == 0):
        raise ValueError("record has zero supervised target tokens")

    supervised_eos = (tokens == eos_id) & (mask == 1)
    response_runs_without_eos = 0
    for record_tokens, record_mask in zip(tokens, mask, strict=True):
        starts = np.flatnonzero(
            (record_mask == 1)
            & np.concatenate(([True], record_mask[:-1] == 0))
        )
        for start in starts:
            end = int(start)
            while end + 1 < len(record_mask) and record_mask[end + 1] == 1:
                end += 1
            if int(record_tokens[end]) != eos_id:
                response_runs_without_eos += 1
    if response_runs_without_eos != int(metadata["truncated_turns"]):
        raise ValueError("truncated response count does not match metadata")

    actual = {
        "records_written": int(tokens.shape[0]),
        "total_tokens": int(tokens.size),
        "assistant_supervised_tokens": int(mask.sum()),
        "padding_tokens": int((tokens == pad_id).sum()),
        "prompt_tokens": int(((mask == 0) & (tokens != pad_id)).sum()),
        "supervised_eos_count": int(supervised_eos.sum()),
    }
    for key, value in actual.items():
        if int(metadata[key]) != value:
            raise ValueError(
                f"metadata mismatch for {key}: {metadata[key]} != {value}"
            )
    train_records = int(metadata["train_records"])
    validation_records = int(metadata["validation_records"])
    if train_records <= 0 or validation_records <= 0:
        raise ValueError("train and validation record splits must be non-empty")
    if train_records + validation_records != len(tokens):
        raise ValueError("record split does not cover the dataset exactly")


def build_metadata(
    *,
    tokens: np.ndarray,
    mask: np.ndarray,
    stats: PackingStats,
    source_path: Path,
    source_sha256: str,
    source_examples_inspected: int,
    selection_stop_example_excluded: int,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    vocabulary_size: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    unk_token_id: int,
    sequence_length: int,
    split_seed: int,
    token_file_sha256: str,
    mask_file_sha256: str,
) -> dict[str, Any]:
    records = int(tokens.shape[0])
    train_records = int(records * 0.95)
    validation_records = records - train_records
    non_padding = stats.prompt_tokens + stats.assistant_supervised_tokens
    assistant_ratio = (
        stats.assistant_supervised_tokens / non_padding if non_padding else 0.0
    )
    return {
        "format_version": FORMAT_VERSION,
        "source_dataset_name": "HuggingFaceH4/ultrachat_200k",
        "source_dataset_path": str(source_path),
        "source_revision": None,
        "source_file_sha256": source_sha256,
        "selection_policy": (
            "deterministic source order; stop before the first source example "
            "that would exceed 5,100,000 fixed-record tokens"
        ),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "vocabulary_size": vocabulary_size,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "unk_token_id": unk_token_id,
        "record_length": sequence_length + 1,
        "sequence_length": sequence_length,
        "total_source_examples_inspected": source_examples_inspected,
        "examples_retained": stats.examples_retained,
        "examples_dropped": stats.examples_dropped,
        "examples_truncated": stats.examples_truncated,
        "malformed_examples": stats.malformed_examples,
        "selection_stop_example_excluded": selection_stop_example_excluded,
        "turns_retained": stats.turns_retained,
        "turns_dropped": stats.turns_dropped,
        "truncated_turns": stats.turns_truncated,
        "records_written": records,
        "total_tokens": int(tokens.size),
        "prompt_tokens": stats.prompt_tokens,
        "assistant_supervised_tokens": stats.assistant_supervised_tokens,
        "supervised_eos_count": stats.supervised_eos_count,
        "padding_tokens": int((tokens == pad_token_id).sum()),
        "assistant_token_ratio": assistant_ratio,
        "train_records": train_records,
        "validation_records": validation_records,
        "split_seed": split_seed,
        "token_dtype": "uint16",
        "mask_dtype": "uint8",
        "token_file_sha256": token_file_sha256,
        "mask_file_sha256": mask_file_sha256,
        "packing_strategy": (
            "fixed 257-token records; canonical User/Assistant turn pairs; "
            "complete responses end in supervised EOS; truncated responses "
            "flush without EOS; packing crosses only supervised EOS boundaries"
        ),
        "stats": asdict(stats),
    }
