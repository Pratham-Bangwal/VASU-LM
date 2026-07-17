"""FineWeb exact and scalable near-duplicate classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .fineweb_index import FineWebDocumentIndex
from .normalization import minhash_signature, normalized_sha256, signature_similarity


class MatchDecision(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    REVIEW = "review"


@dataclass(frozen=True)
class MatchResult:
    match_type: str | None
    decision: MatchDecision
    wikimedia_chunk_id: str
    wikimedia_hash: str
    fineweb_document_id: str | None = None
    fineweb_hash: str | None = None
    similarity: float | None = None
    provenance: dict[str, Any] | None = None


def match_text(
    index: FineWebDocumentIndex,
    text: str,
    *,
    chunk_id: str,
    signature_size: int = 64,
    bands: int = 8,
    reject_threshold: float = 0.90,
    review_threshold: float = 0.75,
) -> MatchResult:
    if not 0 <= review_threshold <= reject_threshold <= 1:
        raise ValueError("matching thresholds must satisfy 0 <= review <= reject <= 1")
    fingerprint = normalized_sha256(text)
    exact = index.exact(fingerprint)
    if exact:
        return _result(exact, chunk_id, fingerprint, "exact", MatchDecision.REJECT, 1.0)
    signature = minhash_signature(text, signature_size=signature_size)
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in index.candidates(signature, bands):
        similarity = signature_similarity(signature, candidate["signature"])
        if best is None or similarity > best[0]:
            best = (similarity, candidate)
    if best and best[0] >= reject_threshold:
        return _result(best[1], chunk_id, fingerprint, "near", MatchDecision.REJECT, best[0])
    if best and best[0] >= review_threshold:
        return _result(best[1], chunk_id, fingerprint, "ambiguous", MatchDecision.REVIEW, best[0])
    return MatchResult(None, MatchDecision.KEEP, chunk_id, fingerprint)


def _result(
    candidate: dict[str, Any],
    chunk_id: str,
    fingerprint: str,
    match_type: str,
    decision: MatchDecision,
    similarity: float,
) -> MatchResult:
    provenance = {
        key: candidate[key]
        for key in (
            "source_id", "source_revision", "source_shard", "source_url",
            "provenance_completeness", "source_content_reference",
        )
    }
    return MatchResult(
        match_type, decision, chunk_id, fingerprint,
        candidate["document_id"], candidate["normalized_sha256"], similarity, provenance,
    )
