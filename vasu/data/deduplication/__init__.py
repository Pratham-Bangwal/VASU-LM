"""Document-level cross-source deduplication utilities."""

from .fineweb_index import FineWebDocumentIndex, build_fineweb_document_index
from .matching import MatchDecision, MatchResult, match_text
from .normalization import NORMALIZATION_VERSION, normalize_for_matching
from .schemas import FineWebIndexConfig, FineWebSource, IndexRecord

__all__ = [
    "FineWebDocumentIndex",
    "FineWebIndexConfig",
    "FineWebSource",
    "IndexRecord",
    "MatchDecision",
    "MatchResult",
    "NORMALIZATION_VERSION",
    "build_fineweb_document_index",
    "match_text",
    "normalize_for_matching",
]
