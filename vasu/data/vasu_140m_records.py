"""Immutable, fixture-testable record contract for the VASU-140M family.

This module packs caller-supplied logical examples. It has no source discovery,
production release, training configuration, or training entry point.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


SCHEMA_ID = "vasu.model-family-records.v1"
FIXTURE_REPORT_SCHEMA_ID = "vasu.model-family-record-fixture-report.v1"
FAMILY_ID = "vasu_140m_v1"
MODEL_CONFIG_SHA256 = (
    "29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059"
)
TOKENIZER_SHA256 = (
    "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
)
VOCAB_SIZE = 32_000
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
CONTEXT_LENGTH = 512
RECORD_WIDTH = CONTEXT_LENGTH + 1
EXPECTED_SPLITS = ("train", "development", "evaluation")
PACKING_ALGORITHM = "complete-example-sequential-input-order-v1"
FROZEN_FIXTURE_REPORT_SHA256 = (
    "7329698aefc3ab4593b1bfba246487a83ed0210b4e3fbe96f72c8bb7b8c8e7e3"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class TextTokenizer(Protocol):
    """Tokenizer surface needed by the fixture compiler."""

    def encode(self, text: str) -> Any: ...


@dataclass(frozen=True)
class LogicalExample:
    """One complete, EOS-terminated supervised example."""

    example_id: str
    split: str
    semantic_sha256: str
    token_ids: tuple[int, ...]
    stored_mask: tuple[int, ...]
    target_start: int


@dataclass(frozen=True)
class ExampleSpan:
    """Location of one complete logical example inside a packed record."""

    example_id: str
    start: int
    end: int
    target_start: int


@dataclass(frozen=True)
class PackedRecord:
    """One independent 513-token record and its aligned stored mask."""

    tokens: np.ndarray
    stored_mask: np.ndarray
    spans: tuple[ExampleSpan, ...]
    used_token_count: int


@dataclass(frozen=True)
class ShiftedTrainingView:
    """The independent 512-position tensors derived from one stored record."""

    input_ids: np.ndarray
    target_ids: np.ndarray
    loss_mask: np.ndarray


def canonical_json(value: object) -> str:
    """Serialize an identity-bearing object deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    """Hash a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def specification() -> dict[str, object]:
    """Return the frozen record and compatibility contract."""

    return {
        "schema_id": SCHEMA_ID,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "context_length": CONTEXT_LENGTH,
        "record_width": RECORD_WIDTH,
        "token_dtype": "uint16",
        "stored_mask_dtype": "uint8",
        "target_derivation": {
            "input": "tokens[:-1]",
            "target": "tokens[1:]",
            "loss_mask": "stored_mask[1:]",
        },
        "tokenizer": {
            "sha256": TOKENIZER_SHA256,
            "vocab_size": VOCAB_SIZE,
            "special_token_ids": {
                "pad": PAD_TOKEN_ID,
                "unk": UNK_TOKEN_ID,
                "bos": BOS_TOKEN_ID,
                "eos": EOS_TOKEN_ID,
            },
        },
        "splits": list(EXPECTED_SPLITS),
        "packing_algorithm": PACKING_ALGORITHM,
        "production_release_created": False,
        "training_authorized": False,
    }


SPECIFICATION_SHA256 = sha256_json(specification())


def compile_text_example(
    *,
    tokenizer: TextTokenizer,
    example_id: str,
    split: str,
    prompt: str,
    response: str,
) -> LogicalExample:
    """Compile text while proving the prompt/response token boundary is stable."""

    if not prompt or not response:
        raise ValueError("prompt and response must be non-empty")
    prefix_encoding = tokenizer.encode(prompt)
    full_encoding = tokenizer.encode(prompt + response)
    prefix_ids = tuple(
        int(token) for token in getattr(prefix_encoding, "ids", prefix_encoding)
    )
    full_ids = tuple(int(token) for token in getattr(full_encoding, "ids", full_encoding))
    if not prefix_ids or len(full_ids) <= len(prefix_ids):
        raise ValueError("tokenized example has no response")
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("tokenizer merged the prompt/response boundary")
    token_ids = (*full_ids, EOS_TOKEN_ID)
    target_start = len(prefix_ids)
    mask = (0,) * target_start + (1,) * (len(token_ids) - target_start)
    semantic = sha256_json({"prompt": prompt, "response": response})
    example = LogicalExample(
        example_id=example_id,
        split=split,
        semantic_sha256=semantic,
        token_ids=token_ids,
        stored_mask=mask,
        target_start=target_start,
    )
    validate_logical_example(example)
    return example


def validate_logical_example(example: LogicalExample) -> None:
    """Fail closed on every logical-example boundary invariant."""

    if not example.example_id:
        raise ValueError("example_id must be non-empty")
    if example.split not in EXPECTED_SPLITS:
        raise ValueError(f"unsupported split: {example.split!r}")
    if not _SHA256_RE.fullmatch(example.semantic_sha256):
        raise ValueError("semantic_sha256 must be lowercase SHA-256")
    if not 0 < len(example.token_ids) <= RECORD_WIDTH:
        raise ValueError(f"{example.example_id!r} exceeds the 513-token record width")
    if len(example.token_ids) != len(example.stored_mask):
        raise ValueError(f"mask length mismatch for {example.example_id!r}")
    if not 0 < example.target_start < len(example.token_ids):
        raise ValueError(f"invalid target_start for {example.example_id!r}")
    if any(type(token) is not int for token in example.token_ids):
        raise ValueError(f"non-integer token for {example.example_id!r}")
    if any(token < 0 or token >= VOCAB_SIZE for token in example.token_ids):
        raise ValueError(f"out-of-vocabulary token for {example.example_id!r}")
    if PAD_TOKEN_ID in example.token_ids:
        raise ValueError(f"content contains PAD for {example.example_id!r}")
    if example.token_ids[-1] != EOS_TOKEN_ID:
        raise ValueError(f"terminal EOS missing for {example.example_id!r}")
    if any(mask not in (0, 1) for mask in example.stored_mask):
        raise ValueError(f"non-binary mask for {example.example_id!r}")
    if any(example.stored_mask[: example.target_start]):
        raise ValueError(f"prompt is supervised for {example.example_id!r}")
    if not all(example.stored_mask[example.target_start :]):
        raise ValueError(f"response or EOS is unsupervised for {example.example_id!r}")


def validate_split_isolation(
    splits: Mapping[str, Sequence[LogicalExample]],
) -> dict[str, int]:
    """Reject missing splits and ID or semantic leakage across all splits."""

    if set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("exact train, development, and evaluation splits are required")
    seen_ids: set[str] = set()
    seen_semantics: set[str] = set()
    counts: dict[str, int] = {}
    for split in EXPECTED_SPLITS:
        if not splits[split]:
            raise ValueError(f"{split} split must be non-empty")
        for example in splits[split]:
            validate_logical_example(example)
            if example.split != split:
                raise ValueError(f"{example.example_id!r} is assigned to the wrong split")
            if example.example_id in seen_ids:
                raise ValueError(f"duplicate example_id: {example.example_id!r}")
            if example.semantic_sha256 in seen_semantics:
                raise ValueError(f"semantic split leakage at {example.example_id!r}")
            seen_ids.add(example.example_id)
            seen_semantics.add(example.semantic_sha256)
        counts[split] = len(splits[split])
    return counts


def pack_split(examples: Sequence[LogicalExample], *, split: str) -> list[PackedRecord]:
    """Pack one split independently without truncating or splitting examples."""

    if split not in EXPECTED_SPLITS or not examples:
        raise ValueError("a supported, non-empty split is required")
    records: list[PackedRecord] = []
    tokens = np.full(RECORD_WIDTH, PAD_TOKEN_ID, dtype=np.uint16)
    mask = np.zeros(RECORD_WIDTH, dtype=np.uint8)
    spans: list[ExampleSpan] = []
    offset = 0

    def finalize() -> None:
        nonlocal tokens, mask, spans, offset
        if spans:
            records.append(PackedRecord(tokens, mask, tuple(spans), offset))
        tokens = np.full(RECORD_WIDTH, PAD_TOKEN_ID, dtype=np.uint16)
        mask = np.zeros(RECORD_WIDTH, dtype=np.uint8)
        spans = []
        offset = 0

    for example in examples:
        validate_logical_example(example)
        if example.split != split:
            raise ValueError(f"{example.example_id!r} cannot be packed into {split}")
        if offset and offset + len(example.token_ids) > RECORD_WIDTH:
            finalize()
        end = offset + len(example.token_ids)
        tokens[offset:end] = np.asarray(example.token_ids, dtype=np.uint16)
        mask[offset:end] = np.asarray(example.stored_mask, dtype=np.uint8)
        spans.append(
            ExampleSpan(
                example_id=example.example_id,
                start=offset,
                end=end,
                target_start=offset + example.target_start,
            )
        )
        offset = end
        if offset == RECORD_WIDTH:
            finalize()
    finalize()
    validate_packed_records(records)
    return records


def validate_packed_records(records: Sequence[PackedRecord]) -> None:
    """Validate stored and shifted-target semantics for packed records."""

    if not records:
        raise ValueError("at least one packed record is required")
    seen_ids: set[str] = set()
    for record in records:
        if record.tokens.shape != (RECORD_WIDTH,) or record.tokens.dtype != np.uint16:
            raise ValueError("tokens must be uint16[513]")
        if (
            record.stored_mask.shape != (RECORD_WIDTH,)
            or record.stored_mask.dtype != np.uint8
        ):
            raise ValueError("stored_mask must be uint8[513]")
        if not 0 < record.used_token_count <= RECORD_WIDTH:
            raise ValueError("invalid used_token_count")
        if np.any(record.tokens >= VOCAB_SIZE):
            raise ValueError("packed record contains out-of-vocabulary token")
        if np.any((record.stored_mask != 0) & (record.stored_mask != 1)):
            raise ValueError("packed record contains non-binary mask")
        if np.any(record.tokens[record.used_token_count :] != PAD_TOKEN_ID):
            raise ValueError("PAD tail contains non-PAD tokens")
        if np.any(record.stored_mask[record.used_token_count :] != 0):
            raise ValueError("PAD tail is supervised")
        expected_start = 0
        for span in record.spans:
            if span.example_id in seen_ids:
                raise ValueError(f"packed duplicate example_id: {span.example_id!r}")
            seen_ids.add(span.example_id)
            if span.start != expected_start or not span.start < span.target_start < span.end:
                raise ValueError(f"invalid packed span for {span.example_id!r}")
            if record.stored_mask[span.start] != 0:
                raise ValueError(f"cross-example target is supervised for {span.example_id!r}")
            if np.any(record.stored_mask[span.start : span.target_start]):
                raise ValueError(f"packed prompt is supervised for {span.example_id!r}")
            if not np.all(record.stored_mask[span.target_start : span.end]):
                raise ValueError(f"packed response is unsupervised for {span.example_id!r}")
            if int(record.tokens[span.end - 1]) != EOS_TOKEN_ID:
                raise ValueError(f"packed EOS missing for {span.example_id!r}")
            expected_start = span.end
        if expected_start != record.used_token_count:
            raise ValueError("packed spans do not cover used tokens")


def shifted_training_view(record: PackedRecord) -> ShiftedTrainingView:
    """Derive one record-local training view without joining adjacent records."""

    validate_packed_records([record])
    return ShiftedTrainingView(
        input_ids=record.tokens[:-1],
        target_ids=record.tokens[1:],
        loss_mask=record.stored_mask[1:],
    )


def build_fixture_report(
    splits: Mapping[str, Sequence[LogicalExample]],
) -> dict[str, object]:
    """Pack and summarize fixture inputs with deterministic lineage hashes."""

    counts = validate_split_isolation(splits)
    packed = {split: pack_split(splits[split], split=split) for split in EXPECTED_SPLITS}
    split_reports: dict[str, object] = {}
    all_logical_identities: dict[str, object] = {}
    for split in EXPECTED_SPLITS:
        examples = splits[split]
        records = packed[split]
        logical_identity = [
            {
                "example_id": item.example_id,
                "semantic_sha256": item.semantic_sha256,
                "token_ids": list(item.token_ids),
                "stored_mask": list(item.stored_mask),
                "target_start": item.target_start,
            }
            for item in examples
        ]
        all_logical_identities[split] = logical_identity
        token_bytes = b"".join(record.tokens.tobytes() for record in records)
        mask_bytes = b"".join(record.stored_mask.tobytes() for record in records)
        split_reports[split] = {
            "logical_example_count": counts[split],
            "logical_sha256": sha256_json(logical_identity),
            "packed_record_count": len(records),
            "token_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "stored_mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
            "used_token_count": sum(record.used_token_count for record in records),
        }
    body: dict[str, object] = {
        "schema_id": FIXTURE_REPORT_SCHEMA_ID,
        "fixture_only": True,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_width": RECORD_WIDTH,
        "specification_sha256": SPECIFICATION_SHA256,
        "logical_input_sha256": sha256_json(all_logical_identities),
        "splits": split_reports,
        "checks": {
            "tokenizer_boundary": True,
            "shifted_mask_alignment": True,
            "pad_tail_masking": True,
            "cross_example_masking": True,
            "cross_record_isolation": True,
            "split_isolation": True,
            "deterministic_rebuild": True,
        },
        "production_release_created": False,
        "training_authorized": False,
    }
    body["report_sha256"] = sha256_json(body)
    validate_fixture_report(body)
    return body


def validate_fixture_report(report: Mapping[str, object]) -> None:
    """Validate a fixture report schema and its self-consistent lineage identity."""

    expected_keys = {
        "schema_id",
        "fixture_only",
        "family_id",
        "model_config_sha256",
        "tokenizer_sha256",
        "record_width",
        "specification_sha256",
        "logical_input_sha256",
        "splits",
        "checks",
        "production_release_created",
        "training_authorized",
        "report_sha256",
    }
    if set(report) != expected_keys:
        raise ValueError("fixture report fields do not match the frozen schema")
    expected_scalars = {
        "schema_id": FIXTURE_REPORT_SCHEMA_ID,
        "fixture_only": True,
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_width": RECORD_WIDTH,
        "specification_sha256": SPECIFICATION_SHA256,
        "production_release_created": False,
        "training_authorized": False,
    }
    for field, expected in expected_scalars.items():
        if report[field] != expected:
            raise ValueError(f"fixture report {field} does not match the specification")
    if not isinstance(report["logical_input_sha256"], str) or not _SHA256_RE.fullmatch(
        report["logical_input_sha256"]
    ):
        raise ValueError("fixture report logical_input_sha256 is invalid")
    checks = report["checks"]
    expected_checks = {
        "tokenizer_boundary",
        "shifted_mask_alignment",
        "pad_tail_masking",
        "cross_example_masking",
        "cross_record_isolation",
        "split_isolation",
        "deterministic_rebuild",
    }
    if (
        not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or any(checks[name] is not True for name in expected_checks)
    ):
        raise ValueError("fixture report checks are incomplete")
    splits = report["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(EXPECTED_SPLITS):
        raise ValueError("fixture report splits do not match the specification")
    expected_split_keys = {
        "logical_example_count",
        "logical_sha256",
        "packed_record_count",
        "token_sha256",
        "stored_mask_sha256",
        "used_token_count",
    }
    for split in EXPECTED_SPLITS:
        split_report = splits[split]
        if not isinstance(split_report, Mapping) or set(split_report) != expected_split_keys:
            raise ValueError(f"fixture report {split} fields are invalid")
        for count_field in (
            "logical_example_count",
            "packed_record_count",
            "used_token_count",
        ):
            value = split_report[count_field]
            if type(value) is not int or value < 1:
                raise ValueError(f"fixture report {split} {count_field} is invalid")
        for hash_field in ("logical_sha256", "token_sha256", "stored_mask_sha256"):
            value = split_report[hash_field]
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"fixture report {split} {hash_field} is invalid")
    reported_hash = report["report_sha256"]
    if not isinstance(reported_hash, str) or not _SHA256_RE.fullmatch(reported_hash):
        raise ValueError("fixture report report_sha256 is invalid")
    body = dict(report)
    del body["report_sha256"]
    if sha256_json(body) != reported_hash:
        raise ValueError("fixture report hash mismatch")


def validate_frozen_fixture_report(report: Mapping[str, object]) -> None:
    """Validate the exact checked-in fixture identity, not only self-consistency."""

    validate_fixture_report(report)
    if report["report_sha256"] != FROZEN_FIXTURE_REPORT_SHA256:
        raise ValueError("fixture report does not match the frozen fixture identity")
