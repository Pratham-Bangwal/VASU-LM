"""Versioned normalization and deterministic word-shingle signatures."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import zlib


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
    """Build a deterministic bounded bottom-k MinHash signature.

    Very long documents are sampled at deterministic, evenly-spaced shingle
    positions.  This bounds CPU while retaining coverage across the document.
    """
    if signature_size < 1:
        raise ValueError("signature size must be positive")
    shingles = word_shingles(text, shingle_size)
    if not shingles:
        return tuple((1 << 64) - 1 for _ in range(signature_size))
    ordered = sorted(shingles)
    maximum_sampled_shingles = max(signature_size * 4, signature_size)
    if len(ordered) > maximum_sampled_shingles:
        ordered = [
            ordered[(index * (len(ordered) - 1)) // (maximum_sampled_shingles - 1)]
            for index in range(maximum_sampled_shingles)
        ]
    maximum = (1 << 64) - 1
    hashes: set[int] = set()
    for shingle in ordered:
        encoded = shingle.encode("utf-8")
        high = zlib.crc32(encoded)
        low = zlib.crc32(encoded, 0x9E3779B9)
        value = (high << 32) | low
        hashes.add(value)
    selected = sorted(hashes)[:signature_size]
    pad_seed = 0
    for value in selected:
        pad_seed ^= value
    while len(selected) < signature_size:
        pad_index = len(selected) + 1
        selected.append(
            (pad_seed ^ (pad_index * 0x9E3779B97F4A7C15)) & maximum
        )
    return tuple(selected)


def signature_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("signatures must be non-empty and have equal length")
    return len(set(left) & set(right)) / len(left)


def bucket_keys(signature: tuple[int, ...], bands: int = DEFAULT_BANDS) -> tuple[str, ...]:
    if bands < 1 or len(signature) % bands:
        raise ValueError("signature length must be divisible by band count")
    keys = []
    for band in range(bands):
        # Partition by stable hash value rather than tuple position. A small
        # edit therefore changes only the affected bands instead of shifting
        # every subsequent bottom-k value.
        values = tuple(value for value in signature if value % bands == band)
        raw = b"".join(value.to_bytes(8, "big") for value in values)
        keys.append(f"{band}:{hashlib.sha256(raw).hexdigest()}")
    return tuple(keys)
