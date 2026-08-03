"""Preallocated inference-only key/value cache for autoregressive decoding."""

from __future__ import annotations

import torch


class PreallocatedKVCache:
    """Store fixed-capacity per-layer K/V tensors and expose populated views.

    The cache is allocated explicitly for one generation request. It is not an
    ``nn.Module`` and therefore cannot appear in a model state dict.
    """

    is_preallocated = True

    def __init__(
        self,
        n_layers: int,
        max_seq_len: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        for name, value in (
            ("n_layers", n_layers),
            ("max_seq_len", max_seq_len),
            ("batch_size", batch_size),
            ("n_heads", n_heads),
            ("head_dim", head_dim),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        shape = (n_layers, 2, batch_size, n_heads, max_seq_len, head_dim)
        self.storage = torch.empty(shape, device=device, dtype=dtype)
        self._sequence_length = 0
        self._next_layer = 0
        self._pending_end: int | None = None

    def _validate_layer(self, layer_idx: int) -> None:
        if not isinstance(layer_idx, int) or not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx must be in [0, {self.n_layers})")

    def _validate_tensor(self, tensor: torch.Tensor) -> None:
        if tensor.ndim != 4:
            raise ValueError(
                "cache tensor must have shape (B, H, T, D); "
                f"got {tuple(tensor.shape)}"
            )
        expected = (self.batch_size, self.n_heads, self.head_dim)
        actual = (tensor.size(0), tensor.size(1), tensor.size(3))
        if actual != expected:
            raise ValueError(
                "cache tensor must have shape (B, H, T, D) matching "
                f"batch/head/head_dim {expected}; got {tuple(tensor.shape)}"
            )
        if tensor.device != self.device:
            raise ValueError("cache tensor device does not match allocation")
        if tensor.dtype != self.dtype:
            raise ValueError("cache tensor dtype does not match allocation")

    def get(
        self, layer_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self._validate_layer(layer_idx)
        length = self.layer_length(layer_idx)
        if length == 0:
            return None, None
        return (
            self.storage[layer_idx, 0, :, :, :length, :],
            self.storage[layer_idx, 1, :, :, :length, :],
        )

    def layer_length(self, layer_idx: int) -> int:
        """Return one layer's populated length without constructing views."""

        self._validate_layer(layer_idx)
        if self._pending_end is not None and layer_idx < self._next_layer:
            return self._pending_end
        return self._sequence_length

    def update(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tokens and return populated prefix views."""

        self._validate_layer(layer_idx)
        self._validate_tensor(key)
        self._validate_tensor(value)
        if key.shape != value.shape:
            raise ValueError("key/value shapes must match")
        start = self._sequence_length
        end = start + key.size(2)
        if end > self.max_seq_len:
            raise ValueError(
                f"cache capacity exceeded: requested {end}, capacity "
                f"{self.max_seq_len}"
            )
        if layer_idx != self._next_layer:
            raise ValueError(
                "preallocated cache layers must update in model order; "
                f"expected layer {self._next_layer}, got {layer_idx}"
            )
        if self._pending_end is None:
            self._pending_end = end
        elif end != self._pending_end:
            raise ValueError("inconsistent per-layer preallocated cache lengths")
        with torch.no_grad():
            self.storage[layer_idx, 0, :, :, start:end, :].copy_(key)
            self.storage[layer_idx, 1, :, :, start:end, :].copy_(value)
        if layer_idx + 1 == self.n_layers:
            self._sequence_length = end
            self._next_layer = 0
            self._pending_end = None
        else:
            self._next_layer = layer_idx + 1
        return (
            self.storage[layer_idx, 0, :, :, :end, :],
            self.storage[layer_idx, 1, :, :, :end, :],
        )

    def reset(self) -> None:
        """Reset logical lengths without reallocating backing storage."""

        self._sequence_length = 0
        self._next_layer = 0
        self._pending_end = None

    clear = reset

    @property
    def sequence_length(self) -> int:
        if self._pending_end is not None:
            raise RuntimeError("cache layers are only partially initialized")
        return self._sequence_length

    @property
    def allocation_bytes(self) -> int:
        return self.storage.numel() * self.storage.element_size()
