"""Validated inference-only key/value cache for autoregressive decoding."""

from __future__ import annotations

import torch


class KVCache:
    """Store per-layer attention keys and values without model parameters.

    This first implementation stores complete tensors and permits dynamic
    concatenation in attention. A future preallocated cache can reduce memory
    copies without changing the public inference semantics.
    """

    def __init__(self, n_layers: int, max_seq_len: int) -> None:
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self._keys: list[torch.Tensor | None] = [None] * n_layers
        self._values: list[torch.Tensor | None] = [None] * n_layers
        self._batch_size: int | None = None
        self._n_heads: int | None = None
        self._head_dim: int | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    def _validate_layer(self, layer_idx: int) -> None:
        if not isinstance(layer_idx, int) or not 0 <= layer_idx < self.n_layers:
            raise IndexError(
                f"layer_idx must be in [0, {self.n_layers}); got {layer_idx}"
            )

    def get(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self._validate_layer(layer_idx)
        return self._keys[layer_idx], self._values[layer_idx]

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store complete K/V tensors for one layer after strict validation."""

        self._validate_layer(layer_idx)
        if key.shape != value.shape:
            raise ValueError(
                f"key/value shapes must match: {key.shape} != {value.shape}"
            )
        if key.ndim != 4:
            raise ValueError("key/value tensors must have shape (B, H, T, D)")
        if key.device != value.device:
            raise ValueError("key/value devices must match")
        if key.dtype != value.dtype:
            raise ValueError("key/value dtypes must match")
        batch_size, n_heads, sequence_length, head_dim = key.shape
        if sequence_length <= 0:
            raise ValueError("cached sequence length must be positive")
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"cache length {sequence_length} exceeds maximum "
                f"{self.max_seq_len}"
            )

        signature = (
            batch_size,
            n_heads,
            head_dim,
            key.device,
            key.dtype,
        )
        expected = (
            self._batch_size,
            self._n_heads,
            self._head_dim,
            self._device,
            self._dtype,
        )
        if self._batch_size is None:
            (
                self._batch_size,
                self._n_heads,
                self._head_dim,
                self._device,
                self._dtype,
            ) = signature
        elif signature != expected:
            raise ValueError(
                "cache batch/head/head-dimension/device/dtype mismatch: "
                f"expected {expected}, got {signature}"
            )

        old_key = self._keys[layer_idx]
        old_length = old_key.size(2) if old_key is not None else None
        if old_length is not None and sequence_length < old_length:
            raise ValueError("cache updates cannot shorten a layer")

        # During one model forward, layers transition one at a time. Other
        # initialized layers may therefore be at either the old or new length,
        # but no third length is valid.
        other_lengths = {
            tensor.size(2)
            for index, tensor in enumerate(self._keys)
            if index != layer_idx and tensor is not None
        }
        allowed_lengths = {sequence_length}
        if old_length is not None:
            allowed_lengths.add(old_length)
        if not other_lengths.issubset(allowed_lengths):
            raise ValueError(
                "inconsistent per-layer cache lengths: "
                f"other={sorted(other_lengths)}, allowed={sorted(allowed_lengths)}"
            )

        self._keys[layer_idx] = key
        self._values[layer_idx] = value

    def reset(self) -> None:
        self._keys = [None] * self.n_layers
        self._values = [None] * self.n_layers
        self._batch_size = None
        self._n_heads = None
        self._head_dim = None
        self._device = None
        self._dtype = None

    # Backward-compatible name for the old cache helper.
    clear = reset

    @property
    def sequence_length(self) -> int:
        initialized = [tensor for tensor in self._keys if tensor is not None]
        if not initialized:
            return 0
        if len(initialized) != self.n_layers:
            raise RuntimeError("cache layers are only partially initialized")
        lengths = {tensor.size(2) for tensor in initialized}
        if len(lengths) != 1:
            raise RuntimeError(
                f"cache layer lengths are inconsistent: {sorted(lengths)}"
            )
        return next(iter(lengths))
