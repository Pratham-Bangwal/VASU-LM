"""Compact, deterministic DataLoader sampling state for exact resume."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch.utils.data import Sampler


@dataclass(frozen=True)
class SamplerPosition:
    """The next batch that has not yet been consumed by training."""

    epoch: int
    next_batch_index: int


class ResumableBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches and store only seed, epoch, and offset.

    A permutation is regenerated from ``seed + epoch`` when an iterator is
    created.  The permutation itself is intentionally not checkpointed, so
    checkpoint size is independent of dataset size.  The trainer must call
    :meth:`mark_batch_consumed` *after* a batch has completed backward
    processing.  That keeps the checkpoint position correct even when
    DataLoader workers have prefetched unconsumed batches.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer.")

        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.drop_last = bool(drop_last)
        self._position = SamplerPosition(epoch=0, next_batch_index=0)

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    @property
    def position(self) -> SamplerPosition:
        return self._position

    @property
    def epoch_complete(self) -> bool:
        return self._position.next_batch_index == self.batches_per_epoch

    def __len__(self) -> int:
        """Return batches remaining in the current epoch."""

        return self.batches_per_epoch - self._position.next_batch_index

    def _epoch_indices(self) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.num_samples, dtype=torch.int64)
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._position.epoch)
        return torch.randperm(self.num_samples, generator=generator)

    def __iter__(self) -> Iterator[list[int]]:
        position = self._position
        indices = self._epoch_indices()
        for batch_index in range(position.next_batch_index, self.batches_per_epoch):
            start = batch_index * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            if self.drop_last and end - start != self.batch_size:
                break
            yield indices[start:end].tolist()

    def mark_batch_consumed(self) -> None:
        """Advance after the trainer, not a DataLoader worker, consumes a batch."""

        if self.epoch_complete:
            raise RuntimeError("Cannot consume a batch after the epoch is complete.")
        self._position = SamplerPosition(
            epoch=self._position.epoch,
            next_batch_index=self._position.next_batch_index + 1,
        )

    def advance_epoch(self) -> None:
        """Move to the next epoch after every current-epoch batch was consumed."""

        if not self.epoch_complete:
            raise RuntimeError("Cannot advance epoch before all batches are consumed.")
        self._position = SamplerPosition(
            epoch=self._position.epoch + 1,
            next_batch_index=0,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return compact, PyTorch-serializable deterministic sampler state."""

        return {
            "format_version": self.FORMAT_VERSION,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "epoch": self._position.epoch,
            "next_batch_index": self._position.next_batch_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a position only when the loader contract matches exactly."""

        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported resumable sampler state format.")
        expected = {
            "num_samples": self.num_samples,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "drop_last": self.drop_last,
        }
        mismatches = [
            name for name, value in expected.items() if state.get(name) != value
        ]
        if mismatches:
            joined = ", ".join(mismatches)
            raise ValueError(
                "Checkpoint sampler state does not match the active DataLoader: "
                f"{joined}. Exact resume is unsafe."
            )
        epoch = state.get("epoch")
        next_batch_index = state.get("next_batch_index")
        if (
            not isinstance(epoch, int)
            or epoch < 0
            or not isinstance(next_batch_index, int)
            or not 0 <= next_batch_index <= self.batches_per_epoch
        ):
            raise ValueError("Checkpoint sampler position is invalid.")
        self._position = SamplerPosition(epoch, next_batch_index)
