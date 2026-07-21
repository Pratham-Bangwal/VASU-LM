from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        data_file="data/processed/tinystories.bin",
        seq_len=256,
        start=0,
        end=None,
        stride=None
    ):

        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.tokens = np.memmap(
            data_file,
            dtype=np.uint16,
            mode="r",
        )

        if end is None:
            end = len(self.tokens)

        self.start = start
        self.end = end

        print(f"Total tokens: {len(self.tokens):,}")
        print(f"Sequence length: {seq_len}")

    def __len__(self):

        return (
            self.end
            - self.start
            - self.seq_len
        ) // self.stride

    def __getitem__(self, idx):
        idx = self.start + idx * self.stride
        
        chunk = self.tokens[
            idx : idx + self.seq_len + 1
        ]

        x = torch.from_numpy(
            chunk[:-1].astype(np.int64)
        )

        y = torch.from_numpy(
            chunk[1:].astype(np.int64)
        )

        return x, y


@dataclass(frozen=True)
class TokenShard:
    """A physical uint16 slice mapped into a contiguous logical stream."""

    path: Path
    physical_start: int
    physical_end: int
    logical_start: int
    logical_end: int
    role: str

    @property
    def token_count(self) -> int:
        return self.physical_end - self.physical_start


class ManifestTokenDataset(Dataset):
    """Read a logical token stream spanning one or more immutable shards.

    The legacy :class:`TextDataset` remains unchanged. This class adds strict
    manifest validation, lazy per-process memmaps, and boundary-spanning reads.
    """

    SUPPORTED_FORMAT_VERSION = 1
    SUPPORTED_DTYPE = "uint16"
    SUPPORTED_SPLITS = {"train", "validation"}

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        sequence_length: int,
        logical_start: int = 0,
        logical_end: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.sequence_length = int(sequence_length)

        if split not in self.SUPPORTED_SPLITS:
            valid = ", ".join(sorted(self.SUPPORTED_SPLITS))
            raise ValueError(f"Unknown split {split!r}; expected one of: {valid}.")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        try:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Manifest is not valid JSON: {self.manifest_path}: {error}"
            ) from error
        if not isinstance(manifest, dict):
            raise ValueError("Manifest root must be a JSON object.")

        self.manifest: dict[str, Any] = manifest
        self._validate_header()
        self._training_shards = self._build_training_shards()
        self._validation_shards = self._build_validation_shards()
        self._validate_validation_isolation()

        self.shards = (
            self._training_shards if split == "train" else self._validation_shards
        )
        self.total_logical_tokens = self.shards[-1].logical_end

        if not isinstance(logical_start, int) or isinstance(logical_start, bool):
            raise TypeError("logical_start must be an integer.")
        if logical_end is not None and (
            not isinstance(logical_end, int) or isinstance(logical_end, bool)
        ):
            raise TypeError("logical_end must be an integer or None.")

        requested_end = self.total_logical_tokens if logical_end is None else logical_end
        if logical_start < 0:
            raise ValueError("logical_start must be non-negative.")
        if requested_end <= logical_start:
            raise ValueError("logical_end must be greater than logical_start.")
        if requested_end > self.total_logical_tokens:
            raise ValueError(
                f"logical_end {requested_end:,} exceeds {split} token capacity "
                f"{self.total_logical_tokens:,}."
            )

        self.logical_start = logical_start
        self.logical_end = requested_end
        available_tokens = self.logical_end - self.logical_start
        if available_tokens < self.sequence_length + 1:
            raise ValueError(
                "Requested logical range does not contain enough tokens for one "
                f"{self.sequence_length}-token sample (need "
                f"{self.sequence_length + 1}, found {available_tokens})."
            )

        self._logical_ends = [shard.logical_end for shard in self.shards]
        self._memmaps: dict[Path, np.memmap] = {}
        self._memmap_pid: int | None = None

    def _validate_header(self) -> None:
        version = self.manifest.get("format_version")
        if version != self.SUPPORTED_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported manifest format_version {version!r}; expected "
                f"{self.SUPPORTED_FORMAT_VERSION}."
            )
        if self.manifest.get("dtype") != self.SUPPORTED_DTYPE:
            raise ValueError("Manifest dtype must be 'uint16'.")

        manifest_sequence_length = self.manifest.get("sequence_length")
        if not isinstance(manifest_sequence_length, int) or manifest_sequence_length <= 0:
            raise ValueError("Manifest sequence_length must be a positive integer.")
        if manifest_sequence_length != self.sequence_length:
            raise ValueError(
                f"Requested sequence_length {self.sequence_length} does not match "
                f"manifest sequence_length {manifest_sequence_length}."
            )

        tokenizer_path = self.manifest.get("tokenizer_path")
        tokenizer_hash = self.manifest.get("tokenizer_sha256")
        if not isinstance(tokenizer_path, str) or not tokenizer_path.strip():
            raise ValueError("Manifest tokenizer_path is required.")
        if not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64:
            raise ValueError("Manifest tokenizer_sha256 must be a SHA-256 hex digest.")
        try:
            int(tokenizer_hash, 16)
        except ValueError as error:
            raise ValueError("Manifest tokenizer_sha256 is not hexadecimal.") from error

        resolved_tokenizer = self._resolve_referenced_path(tokenizer_path)
        if not resolved_tokenizer.exists():
            raise FileNotFoundError(
                f"Manifest tokenizer file not found: {resolved_tokenizer}"
            )
        actual_hash = self._sha256_file(resolved_tokenizer)
        if actual_hash.lower() != tokenizer_hash.lower():
            raise ValueError(
                f"Tokenizer SHA-256 mismatch for {resolved_tokenizer}: expected "
                f"{tokenizer_hash}, found {actual_hash}."
            )
        self.tokenizer_path = resolved_tokenizer
        self.tokenizer_sha256 = tokenizer_hash.lower()

    def _resolve_referenced_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path.resolve()

        # Production manifests use repository-root-relative paths. Synthetic
        # manifests commonly use manifest-relative paths, so support both in a
        # deterministic order.
        working_directory_path = (Path.cwd() / path).resolve()
        if working_directory_path.exists():
            return working_directory_path
        return (self.manifest_path.parent / path).resolve()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_physical_slice(
        self,
        raw: Any,
        *,
        label: str,
        expected_logical_start: int,
    ) -> TokenShard:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a JSON object.")
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"{label}.path is required.")
        path = self._resolve_referenced_path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        byte_size = path.stat().st_size
        if byte_size % np.dtype(np.uint16).itemsize != 0:
            raise ValueError(f"{label} file size is not divisible by two: {path}")
        file_tokens = byte_size // np.dtype(np.uint16).itemsize

        start = raw.get("start_token")
        end = raw.get("end_token")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{label} start_token/end_token must be integers.")
        if start < 0 or end <= start or end > file_tokens:
            raise ValueError(
                f"Invalid physical slice for {label}: [{start}, {end}) with "
                f"file capacity {file_tokens}."
            )

        token_count = end - start
        explicit_start = raw.get("logical_start", expected_logical_start)
        explicit_end = raw.get("logical_end", expected_logical_start + token_count)
        if not isinstance(explicit_start, int) or not isinstance(explicit_end, int):
            raise ValueError(f"{label} logical ranges must be integers.")
        if explicit_start != expected_logical_start:
            relation = "overlap" if explicit_start < expected_logical_start else "gap"
            raise ValueError(
                f"Logical {relation} before {label}: expected start "
                f"{expected_logical_start}, found {explicit_start}."
            )
        if explicit_end != explicit_start + token_count:
            raise ValueError(
                f"{label} logical range length does not match its physical slice."
            )

        return TokenShard(
            path=path,
            physical_start=start,
            physical_end=end,
            logical_start=explicit_start,
            logical_end=explicit_end,
            role=str(raw.get("role", label)),
        )

    def _build_training_shards(self) -> tuple[TokenShard, ...]:
        raw_shards = self.manifest.get("training_shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("Manifest training_shards must be a non-empty list.")
        shards: list[TokenShard] = []
        logical_start = 0
        for index, raw in enumerate(raw_shards):
            shard = self._validate_physical_slice(
                raw,
                label=f"training_shards[{index}]",
                expected_logical_start=logical_start,
            )
            if shard.token_count <= 0:
                raise ValueError(f"training_shards[{index}] must not be empty.")
            shards.append(shard)
            logical_start = shard.logical_end

        declared_total = self.manifest.get("logical_training_tokens")
        if not isinstance(declared_total, int) or declared_total != logical_start:
            raise ValueError(
                "logical_training_tokens does not equal the sum of ordered "
                f"training slices: declared {declared_total!r}, actual {logical_start}."
            )
        return tuple(shards)

    def _build_validation_shards(self) -> tuple[TokenShard, ...]:
        raw_validation = self.manifest.get("validation")
        shard = self._validate_physical_slice(
            raw_validation,
            label="validation",
            expected_logical_start=0,
        )
        if shard.token_count <= 0:
            raise ValueError("Validation slice must not be empty.")
        return (shard,)

    def _validate_validation_isolation(self) -> None:
        validation = self._validation_shards[0]
        for training in self._training_shards:
            if training.path != validation.path:
                continue
            overlap_start = max(training.physical_start, validation.physical_start)
            overlap_end = min(training.physical_end, validation.physical_end)
            if overlap_start < overlap_end:
                raise ValueError(
                    "Validation slice overlaps a training slice in "
                    f"{training.path}: [{overlap_start}, {overlap_end})."
                )

    def _ensure_open(self) -> None:
        process_id = os.getpid()
        if self._memmap_pid == process_id and self._memmaps:
            return
        self._memmaps = {}
        for shard in self.shards:
            if shard.path not in self._memmaps:
                self._memmaps[shard.path] = np.memmap(
                    shard.path,
                    dtype=np.uint16,
                    mode="r",
                )
        self._memmap_pid = process_id

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_memmaps"] = {}
        state["_memmap_pid"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._memmaps = {}
        self._memmap_pid = None

    def _segments_for_read(
        self,
        logical_start: int,
        count: int,
    ) -> list[tuple[TokenShard, int, int]]:
        if logical_start < 0:
            raise ValueError("logical_start must be non-negative.")
        if count <= 0:
            raise ValueError("count must be positive.")
        logical_end = logical_start + count
        if logical_end > self.total_logical_tokens:
            raise IndexError(
                f"Read [{logical_start}, {logical_end}) exceeds {self.split} "
                f"capacity {self.total_logical_tokens}."
            )

        segments: list[tuple[TokenShard, int, int]] = []
        cursor = logical_start
        shard_index = bisect_right(self._logical_ends, cursor)
        while cursor < logical_end:
            shard = self.shards[shard_index]
            take_end = min(logical_end, shard.logical_end)
            physical_start = shard.physical_start + (cursor - shard.logical_start)
            physical_end = physical_start + (take_end - cursor)
            segments.append((shard, physical_start, physical_end))
            cursor = take_end
            shard_index += 1
        return segments

    def describe_read(self, logical_start: int, count: int) -> list[dict[str, Any]]:
        """Return physical composition for diagnostics without reading tokens."""

        return [
            {
                "path": str(shard.path),
                "role": shard.role,
                "physical_start": physical_start,
                "physical_end": physical_end,
                "token_count": physical_end - physical_start,
            }
            for shard, physical_start, physical_end in self._segments_for_read(
                logical_start, count
            )
        ]

    def _read_tokens(self, logical_start: int, count: int) -> np.ndarray:
        segments = self._segments_for_read(logical_start, count)
        self._ensure_open()
        arrays = [
            self._memmaps[shard.path][physical_start:physical_end]
            for shard, physical_start, physical_end in segments
        ]
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays)

    def read_tokens(self, logical_start: int, count: int) -> np.ndarray:
        """Read a logical token range as an independent uint16 array.

        This public, read-only boundary lets deterministic data-preparation
        tooling reuse the validated multi-shard mapping without depending on
        the dataset's private implementation or materializing full shards.
        """
        return np.asarray(
            self._read_tokens(logical_start, count), dtype=np.uint16
        ).copy()

    def __len__(self) -> int:
        available_tokens = self.logical_end - self.logical_start
        return max(0, (available_tokens - 1) // self.sequence_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Dataset index must be an integer.")
        if index < 0:
            raise IndexError("Negative dataset indices are not supported.")
        if index >= len(self):
            raise IndexError(f"Dataset index {index} is out of range.")

        sample_start = self.logical_start + index * self.sequence_length
        tokens = self._read_tokens(sample_start, self.sequence_length + 1)
        # Conversion to int64 copies read-only memmap data into safe torch.long
        # tensors while preserving the one-token target shift.
        token_ids = np.asarray(tokens, dtype=np.int64)
        x = torch.from_numpy(token_ids[:-1].copy())
        y = torch.from_numpy(token_ids[1:].copy())
        return x, y
