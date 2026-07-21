"""Document-level cross-source deduplication utilities."""

from .fineweb_index import (
    FederatedFineWebDocumentIndex,
    FineWebDocumentIndex,
    build_fineweb_document_index,
)
from .extension_recovery import (
    DatasetServerSourceIdClient,
    ExtensionRecoveryAudit,
    HistoricalRecoveryRecord,
    RecoveryError,
    RecoveryResult,
    ScalabilityBlockedError,
    audit_extension_recovery,
    classify_retrieval,
    deterministic_spread_sample,
    historical_extension_fingerprint,
    historical_hash_matches,
    load_historical_records,
    validate_report,
)
from .matching import MatchDecision, MatchResult, match_text
from .normalization import NORMALIZATION_VERSION, normalize_for_matching
from .schemas import FineWebIndexConfig, FineWebSource, IndexRecord

__all__ = [
    "FineWebDocumentIndex",
    "FederatedFineWebDocumentIndex",
    "ExtensionRecoveryAudit",
    "DatasetServerSourceIdClient",
    "FineWebIndexConfig",
    "FineWebSource",
    "IndexRecord",
    "MatchDecision",
    "MatchResult",
    "HistoricalRecoveryRecord",
    "RecoveryError",
    "RecoveryResult",
    "ScalabilityBlockedError",
    "NORMALIZATION_VERSION",
    "build_fineweb_document_index",
    "audit_extension_recovery",
    "classify_retrieval",
    "deterministic_spread_sample",
    "historical_extension_fingerprint",
    "historical_hash_matches",
    "load_historical_records",
    "match_text",
    "normalize_for_matching",
    "validate_report",
]
