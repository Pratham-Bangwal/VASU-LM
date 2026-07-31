"""Full-loss 513-token record contract for VASU-140M base-pretraining text.

This module accepts only caller-supplied, already admitted and normalized text
chunks. It performs no source discovery, acquisition, filtering, publication,
schedule creation, configuration creation, or training.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vasu.data.vasu_140m_records import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    EXPECTED_SPLITS,
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    PAD_TOKEN_ID,
    RECORD_WIDTH,
    SPECIFICATION_SHA256,
    TOKENIZER_SHA256,
    UNK_TOKEN_ID,
    ExampleSpan,
    LogicalExample,
    PackedRecord,
    canonical_json,
    sha256_json,
    shifted_training_view,
    validate_logical_example,
    validate_packed_records,
)


SCHEMA_ID = "vasu_140m_base_text_records_v1"
FIXTURE_REPORT_SCHEMA_ID = "vasu_140m_base_text_record_fixture_report_v1"
MASKING_CONTRACT = "full_loss_except_chunk_boundary_pad_v1"
PACKING_CONTRACT = "complete_chunk_sequential_per_source_split_v1"
_SPECIAL_CONTENT_IDS = frozenset(
    {PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID}
)


class TextTokenizer(Protocol):
    def encode(self, text: str) -> Any: ...


@dataclass(frozen=True)
class BaseTextChunk:
    """One immutable, lineage-bound text chunk ready for full-loss packing."""

    source_id: str
    source_revision: str
    document_id: str
    document_sha256: str
    transformation_id: str
    chunk_id: str
    chunk_index: int
    split: str
    text_sha256: str
    token_ids: tuple[int, ...]
    stored_mask: tuple[int, ...]

    def logical_example(self) -> LogicalExample:
        return LogicalExample(
            example_id=self.chunk_id,
            split=self.split,
            semantic_sha256=self.text_sha256,
            token_ids=self.token_ids,
            stored_mask=self.stored_mask,
            target_start=1,
        )


class BaseChunkStreamValidator:
    """Stateful cross-source/split lineage validation for streaming builders."""

    def __init__(self) -> None:
        self._chunk_ids: set[str] = set()
        self._text_hashes: set[str] = set()
        self._parent_splits: dict[tuple[str, str, str], str] = {}
        self._source_revisions: dict[str, str] = {}
        self._parent_indexes: set[tuple[str, str, str, int]] = set()
        self._counts = {split: 0 for split in EXPECTED_SPLITS}

    def consume(self, chunk: BaseTextChunk) -> None:
        """Validate one chunk and add its immutable identities to the stream."""

        validate_base_text_chunk(chunk)
        if chunk.chunk_id in self._chunk_ids:
            raise ValueError(f"duplicate chunk_id: {chunk.chunk_id!r}")
        if chunk.text_sha256 in self._text_hashes:
            raise ValueError(f"duplicate chunk text: {chunk.chunk_id!r}")
        parent = (chunk.source_id, chunk.source_revision, chunk.document_id)
        prior_split = self._parent_splits.setdefault(parent, chunk.split)
        if prior_split != chunk.split:
            raise ValueError(f"parent document crosses splits: {parent!r}")
        prior_revision = self._source_revisions.setdefault(
            chunk.source_id, chunk.source_revision
        )
        if prior_revision != chunk.source_revision:
            raise ValueError(f"source revision changed for {chunk.source_id!r}")
        parent_index = (*parent, chunk.chunk_index)
        if parent_index in self._parent_indexes:
            raise ValueError(f"duplicate chunk_index within parent: {parent_index!r}")
        self._chunk_ids.add(chunk.chunk_id)
        self._text_hashes.add(chunk.text_sha256)
        self._parent_indexes.add(parent_index)
        self._counts[chunk.split] += 1

    def finish(self) -> dict[str, int]:
        """Return split counts only after every required split was observed."""

        missing = [split for split, count in self._counts.items() if count == 0]
        if missing:
            raise ValueError(f"base chunk stream has empty splits: {missing}")
        return dict(self._counts)


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: str, label: str) -> str:
    _nonempty(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def compile_base_text_chunk(
    *,
    tokenizer: TextTokenizer,
    source_id: str,
    source_revision: str,
    document_id: str,
    document_sha256: str,
    transformation_id: str,
    chunk_id: str,
    chunk_index: int,
    split: str,
    text: str,
) -> BaseTextChunk:
    """Tokenize one normalized chunk without inventing cross-chunk targets."""

    _nonempty(source_id, "source_id")
    _nonempty(source_revision, "source_revision")
    _nonempty(document_id, "document_id")
    _sha256(document_sha256, "document_sha256")
    _nonempty(transformation_id, "transformation_id")
    _nonempty(chunk_id, "chunk_id")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")
    if split not in EXPECTED_SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    _nonempty(text, "text")
    encoding = tokenizer.encode(text)
    content_ids = tuple(int(token) for token in getattr(encoding, "ids", encoding))
    if not content_ids:
        raise ValueError("tokenized base text is empty")
    reserved = sorted(set(content_ids) & _SPECIAL_CONTENT_IDS)
    if reserved:
        raise ValueError(f"base text contains reserved/unknown token IDs: {reserved}")
    token_ids = (*content_ids, EOS_TOKEN_ID)
    if len(token_ids) > RECORD_WIDTH:
        raise ValueError("base text chunk exceeds the 513-token record width")
    chunk = BaseTextChunk(
        source_id=source_id,
        source_revision=source_revision,
        document_id=document_id,
        document_sha256=document_sha256,
        transformation_id=transformation_id,
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        split=split,
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        token_ids=token_ids,
        stored_mask=(0,) + (1,) * (len(token_ids) - 1),
    )
    validate_base_text_chunk(chunk)
    return chunk


def validate_base_text_chunk(chunk: BaseTextChunk) -> None:
    """Validate full-loss semantics and immutable chunk lineage."""

    _nonempty(chunk.source_id, "source_id")
    _nonempty(chunk.source_revision, "source_revision")
    _nonempty(chunk.document_id, "document_id")
    _sha256(chunk.document_sha256, "document_sha256")
    _nonempty(chunk.transformation_id, "transformation_id")
    _nonempty(chunk.chunk_id, "chunk_id")
    if (
        isinstance(chunk.chunk_index, bool)
        or not isinstance(chunk.chunk_index, int)
        or chunk.chunk_index < 0
    ):
        raise ValueError("chunk_index must be a non-negative integer")
    _sha256(chunk.text_sha256, "text_sha256")
    if set(chunk.token_ids[:-1]) & _SPECIAL_CONTENT_IDS:
        raise ValueError("base text content contains reserved/unknown token IDs")
    if chunk.stored_mask != (0,) + (1,) * (len(chunk.token_ids) - 1):
        raise ValueError("base text mask is not full-loss after its chunk boundary")
    validate_logical_example(chunk.logical_example())


def validate_base_split_isolation(
    splits: Mapping[str, Sequence[BaseTextChunk]],
) -> dict[str, int]:
    """Require document-level split isolation and one revision per source."""

    if set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("exact train, development, and evaluation splits are required")
    validator = BaseChunkStreamValidator()
    for split in EXPECTED_SPLITS:
        chunks = splits[split]
        if not chunks:
            raise ValueError(f"{split} split must be non-empty")
        for chunk in chunks:
            if chunk.split != split:
                raise ValueError(f"{chunk.chunk_id!r} is assigned to the wrong split")
            validator.consume(chunk)
    return validator.finish()


def iter_pack_base_chunks(
    chunks: Iterable[BaseTextChunk], *, split: str
) -> Iterator[PackedRecord]:
    """Yield fixed records without materializing a source split in memory."""

    if split not in EXPECTED_SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    tokens = np.full(RECORD_WIDTH, PAD_TOKEN_ID, dtype=np.uint16)
    mask = np.zeros(RECORD_WIDTH, dtype=np.uint8)
    spans: list[ExampleSpan] = []
    offset = 0
    observed = 0

    def finalize() -> PackedRecord:
        return PackedRecord(tokens.copy(), mask.copy(), tuple(spans), offset)

    for chunk in chunks:
        validate_base_text_chunk(chunk)
        if chunk.split != split:
            raise ValueError(f"{chunk.chunk_id!r} cannot be packed into {split}")
        example = chunk.logical_example()
        if offset and offset + len(example.token_ids) > RECORD_WIDTH:
            record = finalize()
            validate_packed_records([record])
            yield record
            tokens.fill(PAD_TOKEN_ID)
            mask.fill(0)
            spans.clear()
            offset = 0
        end = offset + len(example.token_ids)
        tokens[offset:end] = np.asarray(example.token_ids, dtype=np.uint16)
        mask[offset:end] = np.asarray(example.stored_mask, dtype=np.uint8)
        spans.append(
            ExampleSpan(
                example_id=example.example_id,
                start=offset,
                end=end,
                target_start=offset + 1,
            )
        )
        offset = end
        observed += 1
        if offset == RECORD_WIDTH:
            record = finalize()
            validate_packed_records([record])
            yield record
            tokens.fill(PAD_TOKEN_ID)
            mask.fill(0)
            spans.clear()
            offset = 0
    if spans:
        record = finalize()
        validate_packed_records([record])
        yield record
    if observed == 0:
        raise ValueError("at least one base chunk is required")


def pack_base_splits(
    splits: Mapping[str, Sequence[BaseTextChunk]],
) -> dict[str, list[PackedRecord]]:
    """Pack each split independently through the accepted 513-token packer."""

    validate_base_split_isolation(splits)
    packed = {
        split: list(iter_pack_base_chunks(splits[split], split=split))
        for split in EXPECTED_SPLITS
    }
    for records in packed.values():
        validate_packed_records(records)
        for record in records:
            view = shifted_training_view(record)
            if view.loss_mask.shape != (RECORD_WIDTH - 1,):
                raise ValueError("shifted base-training mask has the wrong width")
    return packed


def _chunk_identity(chunk: BaseTextChunk) -> dict[str, object]:
    return {
        "source_id": chunk.source_id,
        "source_revision": chunk.source_revision,
        "document_id": chunk.document_id,
        "document_sha256": chunk.document_sha256,
        "transformation_id": chunk.transformation_id,
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "split": chunk.split,
        "text_sha256": chunk.text_sha256,
        "token_ids": list(chunk.token_ids),
        "stored_mask": list(chunk.stored_mask),
    }


def build_base_fixture_report(
    splits: Mapping[str, Sequence[BaseTextChunk]],
) -> dict[str, object]:
    """Build deterministic, fixture-only evidence for base-text mechanics."""

    counts = validate_base_split_isolation(splits)
    packed = pack_base_splits(splits)
    logical_identity = {
        split: [_chunk_identity(chunk) for chunk in splits[split]]
        for split in EXPECTED_SPLITS
    }
    split_reports: dict[str, object] = {}
    for split in EXPECTED_SPLITS:
        chunks = splits[split]
        records = packed[split]
        token_bytes = b"".join(record.tokens.tobytes() for record in records)
        mask_bytes = b"".join(record.stored_mask.tobytes() for record in records)
        parents = {
            (chunk.source_id, chunk.source_revision, chunk.document_id)
            for chunk in chunks
        }
        source_counts: dict[str, int] = {}
        for chunk in chunks:
            source_counts[chunk.source_id] = source_counts.get(chunk.source_id, 0) + 1
        split_reports[split] = {
            "chunk_count": counts[split],
            "parent_document_count": len(parents),
            "source_chunk_counts": dict(sorted(source_counts.items())),
            "packed_record_count": len(records),
            "real_token_count": sum(len(chunk.token_ids) for chunk in chunks),
            "supervised_target_count": sum(sum(chunk.stored_mask) for chunk in chunks),
            "masked_chunk_boundary_count": len(chunks),
            "padding_token_count": sum(
                RECORD_WIDTH - record.used_token_count for record in records
            ),
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "stored_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        }
    report: dict[str, object] = {
        "schema_id": FIXTURE_REPORT_SCHEMA_ID,
        "fixture_only": True,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_specification_sha256": SPECIFICATION_SHA256,
        "record_width": RECORD_WIDTH,
        "masking_contract": MASKING_CONTRACT,
        "packing_contract": PACKING_CONTRACT,
        "logical_input_sha256": sha256_json(logical_identity),
        "splits": split_reports,
        "checks": {
            "full_loss_after_chunk_boundary": True,
            "chunk_boundary_masking": True,
            "pad_tail_masking": True,
            "cross_record_isolation": True,
            "document_split_isolation": True,
            "source_revision_consistency": True,
            "deterministic_rebuild": True,
        },
        "source_discovered": False,
        "data_acquired": False,
        "production_release_created": False,
        "training_authorized": False,
    }
    report["report_sha256"] = sha256_json(report)
    validate_base_fixture_report(report)
    return report


def validate_base_fixture_report(report: Mapping[str, object]) -> None:
    """Validate fixture evidence without treating it as a production release."""

    expected = {
        "schema_id",
        "fixture_only",
        "family_id",
        "model_config_sha256",
        "tokenizer_sha256",
        "record_specification_sha256",
        "record_width",
        "masking_contract",
        "packing_contract",
        "logical_input_sha256",
        "splits",
        "checks",
        "source_discovered",
        "data_acquired",
        "production_release_created",
        "training_authorized",
        "report_sha256",
    }
    if set(report) != expected:
        raise ValueError("base fixture report fields do not match the schema")
    exact = {
        "schema_id": FIXTURE_REPORT_SCHEMA_ID,
        "fixture_only": True,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_specification_sha256": SPECIFICATION_SHA256,
        "record_width": RECORD_WIDTH,
        "masking_contract": MASKING_CONTRACT,
        "packing_contract": PACKING_CONTRACT,
        "source_discovered": False,
        "data_acquired": False,
        "production_release_created": False,
        "training_authorized": False,
    }
    for field, expected_value in exact.items():
        if report[field] != expected_value:
            raise ValueError(f"base fixture report {field} mismatch")
    _sha256(str(report["logical_input_sha256"]), "logical_input_sha256")
    splits = report["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("base fixture report splits are incomplete")
    split_fields = {
        "chunk_count",
        "parent_document_count",
        "source_chunk_counts",
        "packed_record_count",
        "real_token_count",
        "supervised_target_count",
        "masked_chunk_boundary_count",
        "padding_token_count",
        "token_sha256",
        "stored_mask_sha256",
    }
    for split in EXPECTED_SPLITS:
        item = splits[split]
        if not isinstance(item, Mapping) or set(item) != split_fields:
            raise ValueError(f"base fixture report {split} fields are invalid")
        for field in split_fields - {"source_chunk_counts", "token_sha256", "stored_mask_sha256"}:
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"base fixture report {split} {field} is invalid")
        if item["chunk_count"] < 1 or item["packed_record_count"] < 1:
            raise ValueError(f"base fixture report {split} cannot be empty")
        source_counts = item["source_chunk_counts"]
        if not isinstance(source_counts, Mapping) or not source_counts:
            raise ValueError(f"base fixture report {split} source counts are invalid")
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for key, value in source_counts.items()
        ):
            raise ValueError(f"base fixture report {split} source counts are invalid")
        _sha256(str(item["token_sha256"]), f"{split}.token_sha256")
        _sha256(str(item["stored_mask_sha256"]), f"{split}.stored_mask_sha256")
    checks = report["checks"]
    required_checks = {
        "full_loss_after_chunk_boundary",
        "chunk_boundary_masking",
        "pad_tail_masking",
        "cross_record_isolation",
        "document_split_isolation",
        "source_revision_consistency",
        "deterministic_rebuild",
    }
    if (
        not isinstance(checks, Mapping)
        or set(checks) != required_checks
        or any(checks[name] is not True for name in required_checks)
    ):
        raise ValueError("base fixture report checks are incomplete")
    reported = _sha256(str(report["report_sha256"]), "report_sha256")
    body = dict(report)
    del body["report_sha256"]
    if sha256_json(body) != reported:
        raise ValueError("base fixture report identity mismatch")


def serialized_report_bytes(report: Mapping[str, object]) -> bytes:
    """Return the canonical bytes used for immutable fixture comparison."""

    validate_base_fixture_report(report)
    return (canonical_json(report) + "\n").encode("utf-8")
