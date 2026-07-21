"""Deterministic preparation primitives for fixed-record pretraining mixtures."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from vasu.data.preparation.reporting import atomic_write_json, sha256_file
from vasu.tokenizer.tokenizer import VASUTokenizer
from vasu.training.dataset import ManifestTokenDataset


TOKENIZED_FORMAT = "vasu_wikimedia_tokenized_v1"
MIXTURE_FORMAT = "vasu_pretraining_mixture_v1"
MIXTURE_ARTIFACT_FORMAT = "vasu_fixed_record_mixture_v1"
UINT16_MAX = np.iinfo(np.uint16).max


def canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_replace(temporary: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value.lower()


def _resolve(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def deterministic_parent_split(
    parent_ids: set[str], *, validation_fraction: float, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split parent IDs deterministically without using global RNG state."""
    if len(parent_ids) < 2:
        raise ValueError("at least two parents are required for a train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    validation_count = max(1, min(len(parent_ids) - 1, round(
        len(parent_ids) * validation_fraction
    )))
    ranked = sorted(
        parent_ids,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest(), value
        ),
    )
    validation = frozenset(ranked[:validation_count])
    train_ordered = tuple(sorted(parent_ids.difference(validation)))
    validation_ordered = tuple(sorted(validation))
    return train_ordered, validation_ordered


def prepare_wikimedia_tokenized_split(
    *,
    source_path: Path,
    expected_source_sha256: str,
    tokenizer_path: Path,
    expected_tokenizer_sha256: str,
    output_train_path: Path,
    output_validation_path: Path,
    output_manifest_path: Path,
    validation_fraction: float = 0.05,
    seed: int = 42,
    sequence_length: int = 256,
) -> dict[str, Any]:
    """Tokenize an immutable Wikimedia release after a parent-level split."""
    expected_source_sha256 = _require_hash(
        expected_source_sha256, "expected_source_sha256"
    )
    expected_tokenizer_sha256 = _require_hash(
        expected_tokenizer_sha256, "expected_tokenizer_sha256"
    )
    if sha256_file(source_path) != expected_source_sha256:
        raise ValueError("approved Wikimedia source SHA-256 mismatch")
    if sha256_file(tokenizer_path) != expected_tokenizer_sha256:
        raise ValueError("tokenizer SHA-256 mismatch")

    tokenizer = VASUTokenizer()
    tokenizer.load(str(tokenizer_path))
    vocabulary_size = tokenizer.tokenizer.get_vocab_size()
    if vocabulary_size != 32_000:
        raise ValueError(f"tokenizer vocabulary must be 32000, found {vocabulary_size}")
    eos_id = tokenizer.tokenizer.token_to_id("[EOS]")
    bos_id = tokenizer.tokenizer.token_to_id("[BOS]")
    if eos_id is None:
        raise ValueError("tokenizer does not define [EOS]")

    records: list[dict[str, Any]] = []
    parents: set[str] = set()
    seen_chunks: set[str] = set()
    source_accounted_tokens = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid Wikimedia JSONL line {line_number}") from error
            parent = record.get("parent_document_id")
            chunk_id = record.get("chunk_id")
            text = record.get("cleaned_text")
            token_count = record.get("token_count")
            provenance = record.get("provenance_metadata")
            if not isinstance(parent, str) or not parent:
                raise ValueError(f"line {line_number} has no parent_document_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"line {line_number} has no chunk_id")
            if chunk_id in seen_chunks:
                raise ValueError(f"duplicate Wikimedia chunk_id: {chunk_id}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"line {line_number} has empty cleaned_text")
            if isinstance(token_count, bool) or not isinstance(token_count, int):
                raise ValueError(f"line {line_number} has invalid token_count")
            if not isinstance(provenance, dict) or not provenance:
                raise ValueError(f"line {line_number} lacks provenance_metadata")
            parents.add(parent)
            seen_chunks.add(chunk_id)
            source_accounted_tokens += token_count
            records.append(record)
    if not records:
        raise ValueError("Wikimedia source contains no records")

    train_parents, validation_parents = deterministic_parent_split(
        parents, validation_fraction=validation_fraction, seed=seed
    )
    train_set = frozenset(train_parents)
    validation_set = frozenset(validation_parents)
    if train_set.intersection(validation_set):
        raise AssertionError("parent-level split leaked between train and validation")

    temporary_paths = {
        "train": output_train_path.with_name(output_train_path.name + ".tmp"),
        "validation": output_validation_path.with_name(
            output_validation_path.name + ".tmp"
        ),
    }
    for path in (*temporary_paths.values(),):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
    counters: dict[str, Counter[str]] = {
        "train": Counter(), "validation": Counter()
    }
    min_token_id = vocabulary_size
    max_token_id = -1
    try:
        with temporary_paths["train"].open("wb") as train_handle, temporary_paths[
            "validation"
        ].open("wb") as validation_handle:
            for record in records:
                parent = str(record["parent_document_id"])
                split = "train" if parent in train_set else "validation"
                if parent not in train_set and parent not in validation_set:
                    raise AssertionError(f"parent {parent} was not assigned to a split")
                serialized = str(record["cleaned_text"]).strip() + "\n[EOS]\n"
                token_ids = tokenizer.encode(serialized)
                if not token_ids:
                    raise ValueError(f"tokenization produced no IDs for {record['chunk_id']}")
                local_min = min(token_ids)
                local_max = max(token_ids)
                if local_min < 0 or local_max >= vocabulary_size:
                    raise ValueError(
                        f"token IDs for {record['chunk_id']} exceed vocabulary bounds"
                    )
                min_token_id = min(min_token_id, local_min)
                max_token_id = max(max_token_id, local_max)
                array = np.asarray(token_ids, dtype=np.uint16)
                target = train_handle if split == "train" else validation_handle
                array.tofile(target)
                counters[split]["records"] += 1
                counters[split]["tokens"] += len(token_ids)
            train_handle.flush()
            validation_handle.flush()
            os.fsync(train_handle.fileno())
            os.fsync(validation_handle.fileno())
        if not counters["train"]["tokens"] or not counters["validation"]["tokens"]:
            raise ValueError("both Wikimedia train and validation splits must be non-empty")
        _atomic_replace(temporary_paths["train"], output_train_path)
        _atomic_replace(temporary_paths["validation"], output_validation_path)
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise

    if sha256_file(source_path) != expected_source_sha256:
        raise ValueError("approved Wikimedia source changed during tokenization")
    configuration = {
        "format_version": TOKENIZED_FORMAT,
        "source_path": str(source_path),
        "source_sha256": expected_source_sha256,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": expected_tokenizer_sha256,
        "vocabulary_size": vocabulary_size,
        "eos_handling": "append exact text \\n[EOS]\\n to every retained chunk",
        "bos_handling": "no BOS token appended",
        "eos_token_id": eos_id,
        "bos_token_id": bos_id,
        "dtype": "uint16",
        "sequence_length": sequence_length,
        "validation_fraction": validation_fraction,
        "split_seed": seed,
        "split_strategy": "SHA-256(seed:parent_id) rank; rounded 5% validation",
    }
    manifest: dict[str, Any] = {
        **configuration,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "preparation_configuration_hash": canonical_hash(configuration),
        "total_source_records": len(records),
        "total_source_parents": len(parents),
        "total_source_tokens": source_accounted_tokens,
        "total_binary_tokens": (
            counters["train"]["tokens"] + counters["validation"]["tokens"]
        ),
        "minimum_token_id": min_token_id,
        "maximum_token_id": max_token_id,
        "train": {
            "path": str(output_train_path),
            "sha256": sha256_file(output_train_path),
            "parent_ids": list(train_parents),
            "parent_count": len(train_parents),
            "record_count": counters["train"]["records"],
            "token_count": counters["train"]["tokens"],
        },
        "validation": {
            "path": str(output_validation_path),
            "sha256": sha256_file(output_validation_path),
            "parent_ids": list(validation_parents),
            "parent_count": len(validation_parents),
            "record_count": counters["validation"]["records"],
            "token_count": counters["validation"]["tokens"],
        },
    }
    atomic_write_json(output_manifest_path, manifest)
    return manifest


def build_source_schedule(
    sequence_counts: Mapping[str, int], *, seed: int
) -> tuple[str, ...]:
    """Return a locally seeded exact-count source schedule."""
    if not sequence_counts or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in sequence_counts.values()
    ):
        raise ValueError("sequence_counts must contain non-negative integers")
    schedule = [
        source_id
        for source_id in sorted(sequence_counts)
        for _ in range(sequence_counts[source_id])
    ]
    random.Random(seed).shuffle(schedule)
    return tuple(schedule)


def allocate_sequences(
    total_sequences: int, weights: Mapping[str, float]
) -> dict[str, int]:
    if total_sequences < 1:
        raise ValueError("total_sequences must be positive")
    if not weights or any(
        isinstance(value, bool) or not isinstance(value, (float, int))
        or not math.isfinite(value) or value <= 0
        for value in weights.values()
    ):
        raise ValueError("weights must be finite positive numbers")
    total_weight = sum(float(value) for value in weights.values())
    if not math.isclose(total_weight, 1.0, abs_tol=1e-9, rel_tol=0.0):
        raise ValueError(f"mixture weights sum to {total_weight}, expected 1.0")
    raw = {key: total_sequences * float(value) for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total_sequences - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def build_mixture_artifact(
    config_path: Path, *, repository_root: Path | None = None
) -> dict[str, Any]:
    """Build a deterministic fixed-record mixture without full concatenation."""
    root = (repository_root or Path.cwd()).resolve()
    config = _load_json(config_path)
    if config.get("format_version") != MIXTURE_FORMAT:
        raise ValueError(f"format_version must be {MIXTURE_FORMAT}")
    seed = config.get("seed")
    sequence_length = config.get("sequence_length")
    token_budget = config.get("token_budget")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(sequence_length, int) or sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not isinstance(token_budget, int) or token_budget < sequence_length:
        raise ValueError("token_budget is too small")
    tokenizer_path = _resolve(root, config.get("tokenizer_path"), "tokenizer_path")
    tokenizer_hash = _require_hash(config.get("tokenizer_sha256"), "tokenizer_sha256")
    if sha256_file(tokenizer_path) != tokenizer_hash:
        raise ValueError("mixture tokenizer hash mismatch")

    raw_sources = config.get("sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != 2:
        raise ValueError("this mixture requires exactly two configured sources")
    sources = {source.get("id"): source for source in raw_sources if isinstance(source, dict)}
    if set(sources) != {"fineweb_edu", "wikimedia_factual"}:
        raise ValueError("sources must be fineweb_edu and wikimedia_factual")
    weights = {key: float(value["weight"]) for key, value in sources.items()}

    fineweb = sources["fineweb_edu"]
    fineweb_manifest = _resolve(root, fineweb.get("manifest"), "FineWeb manifest")
    fineweb_manifest_hash = _require_hash(fineweb.get("manifest_sha256"), "FineWeb manifest hash")
    if sha256_file(fineweb_manifest) != fineweb_manifest_hash:
        raise ValueError("FineWeb manifest hash mismatch")
    fineweb_start = fineweb.get("logical_start")
    if not isinstance(fineweb_start, int) or fineweb_start < 0:
        raise ValueError("FineWeb logical_start must be non-negative")
    fineweb_dataset = ManifestTokenDataset(
        fineweb_manifest, "train", sequence_length, logical_start=fineweb_start
    )
    if fineweb_dataset.tokenizer_sha256 != tokenizer_hash:
        raise ValueError("FineWeb tokenizer is incompatible with mixture tokenizer")

    wiki = sources["wikimedia_factual"]
    wiki_path = _resolve(root, wiki.get("path"), "Wikimedia token path")
    wiki_hash = _require_hash(wiki.get("sha256"), "Wikimedia token hash")
    if sha256_file(wiki_path) != wiki_hash:
        raise ValueError("Wikimedia token hash mismatch")
    wiki_manifest_path = _resolve(root, wiki.get("manifest"), "Wikimedia manifest")
    wiki_manifest_hash = _require_hash(
        wiki.get("manifest_sha256"), "Wikimedia manifest hash"
    )
    if sha256_file(wiki_manifest_path) != wiki_manifest_hash:
        raise ValueError("Wikimedia tokenization manifest hash mismatch")
    wiki_manifest = _load_json(wiki_manifest_path)
    if wiki_manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Wikimedia tokenizer is incompatible with mixture tokenizer")
    wiki_tokens = np.memmap(wiki_path, dtype=np.uint16, mode="r")

    sequence_count = token_budget // sequence_length
    actual_budget = sequence_count * sequence_length
    allocations = allocate_sequences(sequence_count, weights)
    schedule = build_source_schedule(allocations, seed=seed)
    schedule_bytes = "".join(f"{source_id}\n" for source_id in schedule).encode("utf-8")
    schedule_hash = hashlib.sha256(schedule_bytes).hexdigest()

    fineweb_required = allocations["fineweb_edu"] * sequence_length + 1
    wiki_required = allocations["wikimedia_factual"] * sequence_length + 1
    if fineweb_start + fineweb_required > fineweb_dataset.total_logical_tokens:
        raise ValueError("FineWeb source lacks enough sequential tokens")
    if wiki_required > len(wiki_tokens):
        raise ValueError("Wikimedia allocation requires replacement sampling")

    output = config.get("output")
    if not isinstance(output, dict):
        raise ValueError("output must be an object")
    output_path = _resolve(root, output.get("path"), "output.path")
    metadata_path = _resolve(root, output.get("metadata"), "output.metadata")
    schedule_path = _resolve(root, output.get("schedule"), "output.schedule")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary_schedule = schedule_path.with_name(schedule_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    temporary_schedule.unlink(missing_ok=True)
    cursors = {"fineweb_edu": 0, "wikimedia_factual": 0}
    minimum_id = UINT16_MAX
    maximum_id = 0
    try:
        with temporary.open("wb") as handle:
            for source_id in schedule:
                cursor = cursors[source_id]
                if source_id == "fineweb_edu":
                    tokens = fineweb_dataset.read_tokens(
                        fineweb_start + cursor * sequence_length,
                        sequence_length + 1,
                    )
                else:
                    start = cursor * sequence_length
                    tokens = np.asarray(
                        wiki_tokens[start : start + sequence_length + 1],
                        dtype=np.uint16,
                    )
                if len(tokens) != sequence_length + 1:
                    raise ValueError(f"short source read from {source_id}")
                minimum_id = min(minimum_id, int(tokens.min()))
                maximum_id = max(maximum_id, int(tokens.max()))
                tokens.tofile(handle)
                cursors[source_id] += 1
            handle.flush()
            os.fsync(handle.fileno())
        with temporary_schedule.open("wb") as handle:
            handle.write(schedule_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_replace(temporary, output_path)
        _atomic_replace(temporary_schedule, schedule_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        temporary_schedule.unlink(missing_ok=True)
        raise
    if maximum_id >= 32_000:
        raise ValueError(f"mixture contains out-of-vocabulary token ID {maximum_id}")

    metadata: dict[str, Any] = {
        "format_version": MIXTURE_ARTIFACT_FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration_path": str(config_path),
        "configuration_sha256": sha256_file(config_path),
        "seed": seed,
        "sequence_length": sequence_length,
        "record_width": sequence_length + 1,
        "dtype": "uint16",
        "requested_token_budget": token_budget,
        "actual_supervised_token_budget": actual_budget,
        "alignment_remainder_tokens": token_budget - actual_budget,
        "record_count": sequence_count,
        "stored_binary_tokens": sequence_count * (sequence_length + 1),
        "minimum_token_id": minimum_id,
        "maximum_token_id": maximum_id,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_hash,
        "selection_schedule_path": str(schedule_path),
        "selection_schedule_sha256": schedule_hash,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "sources": {
            source_id: {
                "weight": weights[source_id],
                "selected_sequences": allocations[source_id],
                "selected_supervised_tokens": (
                    allocations[source_id] * sequence_length
                ),
                "realized_percentage": allocations[source_id] / sequence_count,
            }
            for source_id in sorted(sources)
        },
        "wikimedia_estimated_effective_passes": (
            allocations["wikimedia_factual"] * sequence_length / len(wiki_tokens)
        ),
        "sampling_with_replacement": False,
        "resume_semantics": "record index is deterministic; resume skips experiment_step * batch_size * gradient_accumulation records",
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


class FixedRecordTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-map independent `(sequence_length + 1)` token records."""

    def __init__(
        self,
        path: str | Path,
        sequence_length: int,
        *,
        start_record: int = 0,
        expected_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.sequence_length = sequence_length
        self.record_width = sequence_length + 1
        if expected_sha256 is not None and sha256_file(self.path) != expected_sha256:
            raise ValueError("fixed-record dataset SHA-256 mismatch")
        byte_size = self.path.stat().st_size
        record_bytes = self.record_width * np.dtype(np.uint16).itemsize
        if byte_size == 0 or byte_size % record_bytes:
            raise ValueError("fixed-record binary size is not record-aligned")
        self.total_records = byte_size // record_bytes
        if not 0 <= start_record < self.total_records:
            raise ValueError("start_record is outside the fixed-record dataset")
        self.start_record = start_record
        self._tokens: np.memmap | None = None
        self._pid: int | None = None

    def _ensure_open(self) -> np.memmap:
        if self._tokens is None or self._pid != os.getpid():
            self._tokens = np.memmap(
                self.path,
                dtype=np.uint16,
                mode="r",
                shape=(self.total_records, self.record_width),
            )
            self._pid = os.getpid()
        return self._tokens

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_tokens"] = None
        state["_pid"] = None
        return state

    def __len__(self) -> int:
        return self.total_records - self.start_record

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise IndexError("fixed-record index must be a non-negative integer")
        if index >= len(self):
            raise IndexError("fixed-record index is out of range")
        row = np.asarray(self._ensure_open()[self.start_record + index], dtype=np.int64)
        return torch.from_numpy(row[:-1].copy()), torch.from_numpy(row[1:].copy())
