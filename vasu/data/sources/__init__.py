"""Public API for VASU dataset provenance and approval records."""

from pathlib import Path
from typing import Any

from .schemas import APPROVAL_STATUSES, DataSourceRecord, SourceRegistry
from .validation import (
    SourceRegistryValidationError,
    validate_manifest_sources,
    validate_registry,
    validate_source_record,
)


# Keep registry imports lazy so ``python -m vasu.data.sources.registry`` can
# execute without the package pre-importing the target module through __init__.
def load_source_record(path: Path) -> DataSourceRecord:
    from .registry import load_source_record as implementation

    return implementation(path)


def load_source_records(path: Path) -> tuple[DataSourceRecord, ...]:
    from .registry import load_source_records as implementation

    return implementation(path)


def load_source_registry(directory: Path) -> SourceRegistry:
    from .registry import load_source_registry as implementation

    return implementation(directory)


def get_source(registry: SourceRegistry, source_id: str) -> DataSourceRecord:
    from .registry import get_source as implementation

    return implementation(registry, source_id)


def source_record_to_dict(record: DataSourceRecord) -> dict[str, Any]:
    from .registry import source_record_to_dict as implementation

    return implementation(record)

__all__ = [
    "APPROVAL_STATUSES",
    "DataSourceRecord",
    "SourceRegistry",
    "SourceRegistryValidationError",
    "get_source",
    "load_source_record",
    "load_source_records",
    "load_source_registry",
    "source_record_to_dict",
    "validate_manifest_sources",
    "validate_registry",
    "validate_source_record",
]
