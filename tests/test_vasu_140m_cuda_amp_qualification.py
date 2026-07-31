from pathlib import Path

import pytest
import torch

from scripts.qualify_vasu_140m_cuda_amp import (
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    require_cuda,
    select_amp_dtype,
    sha256_file,
    synthetic_tokens,
)


def test_synthetic_tokens_are_deterministic_and_have_the_full_boundary() -> None:
    first = synthetic_tokens(32, torch.device("cpu"))
    second = synthetic_tokens(32, torch.device("cpu"))
    assert BATCH_SIZE == 1
    assert SEQUENCE_LENGTH == 512
    assert all(torch.equal(left, right) for left, right in zip(first, second))


def test_amp_dtype_prefers_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda _: True)
    assert select_amp_dtype(0) == (torch.bfloat16, None)


def test_amp_dtype_has_explicit_fp16_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda _: False)
    assert select_amp_dtype(0) == (torch.float16, "cuda_bf16_not_supported")


def test_cuda_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires available CUDA"):
        require_cuda(0)


def test_file_hash_is_content_sensitive(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"one")
    first = sha256_file(path)
    path.write_bytes(b"two")
    assert first != sha256_file(path)
