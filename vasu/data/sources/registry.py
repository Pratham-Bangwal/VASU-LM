"""JSON loading and lookup for the VASU data-source registry."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import DataSourceRecord, SourceRegistry
from .validation import validate_registry, validate_source_record


RECORD_FIELDS = frozenset(DataSourceRecord.__dataclass_fields__)
LIST_FIELDS = frozenset(
    {
        "supported_domains",
        "quality_risks",
        "required_quality_filters",
        "deduplication_requirements",
        "contamination_risks",
    }
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Source record does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Source record is not valid JSON at line {error.lineno}, "
            f"column {error.colno}: {path}"
        ) from error


def _record_from_mapping(payload: Mapping[str, Any], location: str) -> DataSourceRecord:
    missing = sorted(RECORD_FIELDS - set(payload))
    unknown = sorted(set(payload) - RECORD_FIELDS)
    if missing:
        raise ValueError(f"{location} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{location} has unknown fields: {unknown}")
    values = dict(payload)
    for field in LIST_FIELDS:
        if not isinstance(values[field], list):
            raise ValueError(f"{location}.{field} must be a JSON list")
        values[field] = tuple(values[field])
    record = DataSourceRecord(**values)
    validate_source_record(record)
    return record


def load_source_record(path: Path) -> DataSourceRecord:
    """Load one record; family bundles must be loaded through the registry."""
    payload = _read_json(Path(path))
    if not isinstance(payload, Mapping) or "records" in payload:
        raise ValueError(f"Expected one source-record JSON object: {path}")
    return _record_from_mapping(payload, str(path))


def _load_records(path: Path) -> tuple[DataSourceRecord, ...]:
    payload = _read_json(path)
    if isinstance(payload, Mapping) and set(payload) == {"records"}:
        records = payload["records"]
        if not isinstance(records, list) or not records:
            raise ValueError(f"{path}.records must be a non-empty JSON list")
        result = []
        for index, item in enumerate(records):
            if not isinstance(item, Mapping):
                raise ValueError(f"{path}.records[{index}] must be a JSON object")
            result.append(_record_from_mapping(item, f"{path}.records[{index}]"))
        return tuple(result)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a source record or records bundle: {path}")
    return (_record_from_mapping(payload, str(path)),)


def load_source_registry(directory: Path) -> SourceRegistry:
    registry_directory = Path(directory)
    if not registry_directory.is_dir():
        raise FileNotFoundError(
            f"Source registry directory does not exist: {registry_directory}"
        )
    paths = sorted(registry_directory.glob("*.json"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"Source registry contains no JSON records: {registry_directory}")
    records = tuple(record for path in paths for record in _load_records(path))
    registry = SourceRegistry(records=records)
    validate_registry(registry)
    return registry


def get_source(registry: SourceRegistry, source_id: str) -> DataSourceRecord:
    for record in registry.records:
        if record.source_id == source_id:
            return record
    valid = ", ".join(sorted(registry.source_ids))
    raise KeyError(f"Unknown source_id {source_id!r}; available source IDs: {valid}")


def source_record_to_dict(record: DataSourceRecord) -> dict[str, Any]:
    payload = asdict(record)
    for field in LIST_FIELDS:
        payload[field] = list(payload[field])
    return payload

