"""Versioned normalization and deterministic word-shingle signatures."""

from __future__ import annotations

import hashlib
import re
import unicodedata


NORMALIZATION_VERSION = "vasu_cross_source_nfc_casefold_ws_v1"
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
DEFAULT_SHINGLE_SIZE = 5
DEFAULT_SIGNATURE_SIZE = 64
DEFAULT_BANDS = 8


def normalize_for_matching(text: str) -> str:
    """Normalize without lossy transliteration or punctuation deletion."""
    if not isinstance(text, str):
        raise TypeError("matching input must be text")
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_for_matching(text).encode("utf-8")).hexdigest()


def word_shingles(text: str, size: int = DEFAULT_SHINGLE_SIZE) -> frozenset[str]:
    if size < 1:
        raise ValueError("shingle size must be positive")
    words = WORD_PATTERN.findall(normalize_for_matching(text))
    if not words:
        return frozenset()
    if len(words) < size:
        return frozenset({"\u241f".join(words)})
    return frozenset(
        "\u241f".join(words[index : index + size])
        for index in range(len(words) - size + 1)
    )


def jaccard_similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def minhash_signature(
    text: str,
    *,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
    signature_size: int = DEFAULT_SIGNATURE_SIZE,
) -> tuple[int, ...]:
    if signature_size < 1:
        raise ValueError("signature size must be positive")
    shingles = word_shingles(text, shingle_size)
    if not shingles:
        return tuple((1 << 64) - 1 for _ in range(signature_size))
    signature: list[int] = []
    for seed in range(signature_size):
        prefix = seed.to_bytes(4, "big")
        signature.append(
            min(
                int.from_bytes(
                    hashlib.blake2b(prefix + shingle.encode("utf-8"), digest_size=8).digest(),
                    "big",
                )
                for shingle in shingles
            )
        )
    return tuple(signature)


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("signatures must be non-empty and have equal length")
    return sum(a == b for a, b in zip(left, right)) / len(left)


def bucket_keys(signature: tuple[int, ...], bands: int = DEFAULT_BANDS) -> tuple[str, ...]:
    if bands < 1 or len(signature) % bands:
        raise ValueError("signature length must be divisible by band count")
    rows = len(signature) // bands
    keys = []
    for band in range(bands):
        values = signature[band * rows : (band + 1) * rows]
        raw = b"".join(value.to_bytes(8, "big") for value in values)
        keys.append(f"{band}:{hashlib.sha256(raw).hexdigest()}")
    return tuple(keys)
