"""Deterministic two-pass qualification for a future VASU-140M base release.

The module writes only caller-selected scratch directories. It never publishes
production data, creates a schedule, creates a training configuration, or
authorizes training.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from typing import Any, Protocol

import numpy as np

from vasu.data.vasu_140m_base_records import (
    BaseChunkStreamValidator,
    BaseTextChunk,
    MASKING_CONTRACT,
    PACKING_CONTRACT,
    iter_pack_base_chunks,
    validate_base_text_chunk,
)
from vasu.data.vasu_140m_records import (
    EOS_TOKEN_ID,
    EXPECTED_SPLITS,
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    PAD_TOKEN_ID,
    RECORD_WIDTH,
    SPECIFICATION_SHA256,
    TOKENIZER_SHA256,
    canonical_json,
    sha256_json,
)


SPEC_SCHEMA_ID = "vasu_140m_base_release_qualification_spec_v1"
INTERNAL_MANIFEST_SCHEMA_ID = "vasu_140m_base_release_internal_manifest_v1"
EXTERNAL_TEMPLATE_SCHEMA_ID = "vasu_140m_base_release_external_template_v1"
QUALIFICATION_SCHEMA_ID = "vasu_140m_base_release_qualification_v1"
SOURCE_EVIDENCE_SCHEMA_ID = "vasu_140m_base_source_release_evidence_v1"
PRODUCTION_RELEASE_PATH = "data/processed/vasu_140m/base_pretraining/v1"
PRODUCTION_MANIFEST_PATH = "data/manifests/vasu_140m/base_pretraining/v1.json"
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_EVIDENCE_KINDS = (
    "admission",
    "acquisition_receipt",
    "raw_inventory",
    "normalization_manifest",
    "quarantine_decision",
    "deduplication_evidence",
    "split_assignment",
    "selection_index",
)
_EVIDENCE_STATUSES = {
    "admission": "approved",
    "acquisition_receipt": "completed",
    "raw_inventory": "complete",
    "normalization_manifest": "complete",
    "quarantine_decision": "accepted",
    "deduplication_evidence": "complete",
    "split_assignment": "complete",
    "selection_index": "complete",
}


class RoundTripTokenizer(Protocol):
    def encode(self, text: str) -> Any: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class QualifiedBaseChunk:
    """One compiled chunk plus normalized text for complete round-trip checks."""

    chunk: BaseTextChunk
    normalized_text: str


@dataclass(frozen=True)
class BuiltPass:
    """Validated identity evidence for one scratch construction pass."""

    root: Path
    internal_manifest: Mapping[str, object]
    external_template: Mapping[str, object]
    artifact_evidence: Mapping[str, Mapping[str, object]]
    pass_sha256: str


StreamFactory = Callable[[str, str], Iterable[QualifiedBaseChunk]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_json(path: Path, value: object) -> None:
    payload = _json_bytes(value)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_line(handle: Any, value: object) -> None:
    handle.write(canonical_json(value).encode("utf-8") + b"\n")


def _exact(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{label} fields mismatch: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _commit(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 40 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git identity")
    return text


def _positive_int(value: object, label: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0 if zero else value <= 0:
        raise ValueError(f"{label} must be {'non-negative' if zero else 'positive'}")
    return value


def _safe_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return text


def _binding(value: object, label: str) -> tuple[str, str]:
    binding = _mapping(value, label)
    _exact(binding, {"path", "sha256"}, label)
    return _safe_path(binding["path"], f"{label}.path"), _sha(
        binding["sha256"], f"{label}.sha256"
    )


def source_evidence_identity(evidence: Mapping[str, object]) -> str:
    body = dict(evidence)
    body.pop("evidence_sha256", None)
    return sha256_json(body)


def validate_source_evidence_envelope(
    evidence: Mapping[str, object],
    *,
    kind: str,
    source_id: str,
    source_revision: str,
    fixture_only: bool,
) -> None:
    """Validate one semantic evidence envelope before any source bytes are read."""

    _exact(
        evidence,
        {
            "schema_id",
            "kind",
            "source_id",
            "source_revision",
            "subject",
            "decision",
            "status",
            "complete",
            "fixture_only",
            "additional_acquisition_authorized",
            "publication_authorized",
            "training_authorized",
            "evidence_sha256",
        },
        f"source evidence {kind}",
    )
    if kind not in _EVIDENCE_STATUSES:
        raise ValueError("source evidence kind is unsupported")
    expected = {
        "schema_id": SOURCE_EVIDENCE_SCHEMA_ID,
        "kind": kind,
        "source_id": source_id,
        "source_revision": source_revision,
        "status": _EVIDENCE_STATUSES[kind],
        "complete": True,
        "fixture_only": fixture_only,
        "additional_acquisition_authorized": False,
        "publication_authorized": False,
        "training_authorized": False,
    }
    for field, value in expected.items():
        if evidence[field] != value:
            raise ValueError(f"source evidence {kind} {field} mismatch")
    _binding(evidence["subject"], f"source evidence {kind}.subject")
    decision = evidence["decision"]
    if kind in {"admission", "quarantine_decision"}:
        _binding(decision, f"source evidence {kind}.decision")
    elif decision is not None:
        raise ValueError(f"source evidence {kind}.decision must be null")
    if _sha(
        evidence["evidence_sha256"], f"source evidence {kind}.evidence_sha256"
    ) != source_evidence_identity(evidence):
        raise ValueError(f"source evidence {kind} identity mismatch")


def specification_identity(specification: Mapping[str, object]) -> str:
    body = dict(specification)
    body.pop("specification_sha256", None)
    return sha256_json(body)


def _validate_count_table(value: object, label: str) -> dict[str, dict[str, int]]:
    table = _mapping(value, label)
    if set(table) != set(EXPECTED_SPLITS):
        raise ValueError(f"{label} must contain every split exactly once")
    result: dict[str, dict[str, int]] = {}
    for split in EXPECTED_SPLITS:
        counts = _mapping(table[split], f"{label}.{split}")
        _exact(
            counts,
            {"chunk_count", "real_token_count", "supervised_target_count"},
            f"{label}.{split}",
        )
        chunks = _positive_int(counts["chunk_count"], f"{label}.{split}.chunk_count")
        real = _positive_int(
            counts["real_token_count"], f"{label}.{split}.real_token_count"
        )
        supervised = _positive_int(
            counts["supervised_target_count"],
            f"{label}.{split}.supervised_target_count",
        )
        if supervised != real - chunks:
            raise ValueError(
                f"{label}.{split} does not match full-loss boundary masking"
            )
        result[split] = {
            "chunk_count": chunks,
            "real_token_count": real,
            "supervised_target_count": supervised,
        }
    return result


def validate_release_specification(specification: Mapping[str, object]) -> None:
    """Validate one immutable, non-authorizing qualification specification."""

    _exact(
        specification,
        {
            "schema_id",
            "release_id",
            "qualification_scope",
            "repository_commit",
            "family_id",
            "model_config_sha256",
            "tokenizer",
            "record_specification_sha256",
            "record_implementation",
            "builder_implementation",
            "release_directory",
            "external_manifest_path",
            "source_order",
            "sources",
            "evaluation_inventories",
            "minimum_free_bytes",
            "publication_authorized",
            "training_authorized",
            "specification_sha256",
        },
        "base release specification",
    )
    if specification["schema_id"] != SPEC_SCHEMA_ID:
        raise ValueError("base release specification schema mismatch")
    _string(specification["release_id"], "release_id")
    if specification["qualification_scope"] not in {"fixture", "production"}:
        raise ValueError("qualification_scope must be fixture or production")
    _commit(specification["repository_commit"], "repository_commit")
    if specification["family_id"] != FAMILY_ID:
        raise ValueError("base release family mismatch")
    if (
        _sha(specification["model_config_sha256"], "model_config_sha256")
        != MODEL_CONFIG_SHA256
    ):
        raise ValueError("base release model configuration mismatch")
    if _binding(specification["tokenizer"], "tokenizer")[1] != TOKENIZER_SHA256:
        raise ValueError("base release tokenizer identity mismatch")
    if (
        _sha(
            specification["record_specification_sha256"],
            "record_specification_sha256",
        )
        != SPECIFICATION_SHA256
    ):
        raise ValueError("base release record specification mismatch")
    _binding(specification["record_implementation"], "record_implementation")
    _binding(specification["builder_implementation"], "builder_implementation")
    if specification["release_directory"] != PRODUCTION_RELEASE_PATH:
        raise ValueError("base release production directory mismatch")
    if specification["external_manifest_path"] != PRODUCTION_MANIFEST_PATH:
        raise ValueError("base release manifest path mismatch")
    if specification["publication_authorized"] is not False:
        raise ValueError("publication_authorized must be false")
    if specification["training_authorized"] is not False:
        raise ValueError("training_authorized must be false")
    if (
        _positive_int(specification["minimum_free_bytes"], "minimum_free_bytes")
        < 10_000_000_000
    ):
        raise ValueError("minimum_free_bytes must reserve at least 10 GB")

    order = specification["source_order"]
    sources = specification["sources"]
    if not isinstance(order, list) or not order:
        raise ValueError("source_order must be a non-empty list")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty list")
    if len(order) != len(set(order)) or any(
        not isinstance(item, str) or not _SOURCE_ID.fullmatch(item) for item in order
    ):
        raise ValueError("source_order contains invalid or duplicate source IDs")
    observed: list[str] = []
    bound_paths: set[str] = set()
    for index, raw in enumerate(sources):
        source = _mapping(raw, f"source {index}")
        _exact(
            source,
            {
                "source_id",
                "source_revision",
                "evidence",
                "expected_counts",
                "no_replacement",
            },
            f"source {index}",
        )
        source_id = _string(source["source_id"], f"source {index}.source_id")
        if not _SOURCE_ID.fullmatch(source_id):
            raise ValueError(f"source {index}.source_id is unsafe")
        observed.append(source_id)
        _string(source["source_revision"], f"source {index}.source_revision")
        if source["no_replacement"] is not True:
            raise ValueError("every base source must prohibit replacement")
        _validate_count_table(
            source["expected_counts"], f"source {source_id}.expected_counts"
        )
        evidence = _mapping(source["evidence"], f"source {source_id}.evidence")
        _exact(evidence, set(_EVIDENCE_KINDS), f"source {source_id}.evidence")
        for kind in _EVIDENCE_KINDS:
            path, _ = _binding(evidence[kind], f"source {source_id}.evidence.{kind}")
            canonical = path.casefold()
            if canonical in bound_paths:
                raise ValueError("source evidence paths must be globally unique")
            bound_paths.add(canonical)
    if observed != order:
        raise ValueError("source list must match source_order exactly")

    inventories = specification["evaluation_inventories"]
    if not isinstance(inventories, list) or not inventories:
        raise ValueError("evaluation_inventories must be non-empty")
    for index, raw in enumerate(inventories):
        path, _ = _binding(raw, f"evaluation inventory {index}")
        canonical = path.casefold()
        if canonical in bound_paths:
            raise ValueError("all evidence paths must be globally unique")
        bound_paths.add(canonical)
    if _sha(
        specification["specification_sha256"], "specification_sha256"
    ) != specification_identity(specification):
        raise ValueError("base release specification identity mismatch")


def validate_release_specification_files(
    specification: Mapping[str, object],
    repository_root: Path,
    *,
    runtime_commit: str,
) -> None:
    """Verify every bound repository file and protected output absence."""

    validate_release_specification(specification)
    if _commit(runtime_commit, "runtime_commit") != specification["repository_commit"]:
        raise ValueError("runtime commit does not match the release specification")
    root = repository_root.resolve()
    bindings = [
        (*_binding(specification["tokenizer"], "tokenizer"), "tokenizer"),
        (
            *_binding(specification["record_implementation"], "record_implementation"),
            "record implementation",
        ),
        (
            *_binding(
                specification["builder_implementation"], "builder_implementation"
            ),
            "builder implementation",
        ),
    ]
    for source in specification["sources"]:
        source_id = str(source["source_id"])
        for kind in _EVIDENCE_KINDS:
            bindings.append(
                (
                    *_binding(source["evidence"][kind], f"source {source_id}.{kind}"),
                    f"source {source_id} {kind}",
                )
            )
    for index, inventory in enumerate(specification["evaluation_inventories"]):
        bindings.append(
            (
                *_binding(inventory, f"evaluation inventory {index}"),
                f"evaluation inventory {index}",
            )
        )
    for relative, expected, label in bindings:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{label} is missing or escapes the repository")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} file identity mismatch")
    nested_bindings: list[tuple[str, str, str]] = []
    for source in specification["sources"]:
        source_id = str(source["source_id"])
        source_revision = str(source["source_revision"])
        for kind in _EVIDENCE_KINDS:
            envelope_relative, _ = _binding(
                source["evidence"][kind], f"source {source_id}.{kind}"
            )
            envelope_path = (root / envelope_relative).resolve()
            try:
                envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(
                    f"source {source_id} {kind} envelope is invalid JSON"
                ) from error
            if not isinstance(envelope, Mapping):
                raise ValueError(
                    f"source {source_id} {kind} envelope must be an object"
                )
            validate_source_evidence_envelope(
                envelope,
                kind=kind,
                source_id=source_id,
                source_revision=source_revision,
                fixture_only=specification["qualification_scope"] == "fixture",
            )
            nested_bindings.append(
                (
                    *_binding(
                        envelope["subject"], f"source {source_id}.{kind}.subject"
                    ),
                    f"source {source_id} {kind} subject",
                )
            )
            if envelope["decision"] is not None:
                nested_bindings.append(
                    (
                        *_binding(
                            envelope["decision"],
                            f"source {source_id}.{kind}.decision",
                        ),
                        f"source {source_id} {kind} decision",
                    )
                )
    for relative, expected, label in nested_bindings:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"{label} is missing or escapes the repository")
        if sha256_file(path) != expected:
            raise ValueError(f"{label} file identity mismatch")
    for field in ("release_directory", "external_manifest_path"):
        path = (root / _safe_path(specification[field], field)).resolve()
        if not path.is_relative_to(root) or path.exists():
            raise ValueError(f"{field} must be absent inside the repository")


def _source_specs(
    specification: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    return {str(source["source_id"]): source for source in specification["sources"]}


def _token_ids(value: Any) -> tuple[int, ...]:
    return tuple(int(token) for token in getattr(value, "ids", value))


def _validate_round_trip(
    item: QualifiedBaseChunk,
    tokenizer: RoundTripTokenizer,
    *,
    source_id: str,
    split: str,
) -> None:
    chunk = item.chunk
    validate_base_text_chunk(chunk)
    if chunk.source_id != source_id or chunk.split != split:
        raise ValueError("stream returned a chunk for the wrong source or split")
    if (
        hashlib.sha256(item.normalized_text.encode("utf-8")).hexdigest()
        != chunk.text_sha256
    ):
        raise ValueError("normalized text does not match chunk text identity")
    encoded = _token_ids(tokenizer.encode(item.normalized_text))
    if (*encoded, EOS_TOKEN_ID) != chunk.token_ids:
        raise ValueError("encode result does not match compiled chunk tokens")
    decoded = tokenizer.decode(list(encoded))
    if _token_ids(tokenizer.encode(decoded)) != encoded:
        raise ValueError("chunk fails encode/decode/encode round trip")


def _artifact(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _relative_artifact_path(source_id: str, split: str, kind: str) -> str:
    suffix = "jsonl" if kind == "lineage" else "bin"
    return f"{source_id}/{split}.{kind}.{suffix}"


def _build_pass(
    *,
    specification: Mapping[str, object],
    tokenizer: RoundTripTokenizer,
    stream_factory: StreamFactory,
    scratch_root: Path,
) -> BuiltPass:
    scratch_root.mkdir(parents=False)
    global_validator = BaseChunkStreamValidator()
    source_specs = _source_specs(specification)
    split_reports: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    logical_digest = hashlib.sha256()
    total_chunks = 0
    total_real = 0
    total_supervised = 0
    decoded_round_trips = 0
    source_totals: dict[str, dict[str, int]] = {}

    for source_id in specification["source_order"]:
        source = source_specs[source_id]
        source_root = scratch_root / source_id
        source_root.mkdir()
        source_reports: dict[str, object] = {}
        expected_table = source["expected_counts"]
        for split in EXPECTED_SPLITS:
            token_path = source_root / f"{split}.tokens.bin"
            mask_path = source_root / f"{split}.mask.bin"
            lineage_path = source_root / f"{split}.lineage.jsonl"
            pending: dict[str, QualifiedBaseChunk] = {}
            observed_chunks = 0
            observed_real = 0
            observed_supervised = 0
            record_count = 0

            def checked_stream() -> Iterable[BaseTextChunk]:
                nonlocal observed_chunks, observed_real, observed_supervised
                nonlocal decoded_round_trips
                for item in stream_factory(source_id, split):
                    if not isinstance(item, QualifiedBaseChunk):
                        raise ValueError(
                            "stream factory must yield QualifiedBaseChunk values"
                        )
                    _validate_round_trip(
                        item, tokenizer, source_id=source_id, split=split
                    )
                    chunk = item.chunk
                    if chunk.source_revision != source["source_revision"]:
                        raise ValueError(
                            "chunk source revision does not match the specification"
                        )
                    global_validator.consume(chunk)
                    if chunk.chunk_id in pending:
                        raise ValueError("duplicate pending chunk identity")
                    pending[chunk.chunk_id] = item
                    row = {
                        "source_id": chunk.source_id,
                        "source_revision": chunk.source_revision,
                        "document_id": chunk.document_id,
                        "document_sha256": chunk.document_sha256,
                        "transformation_id": chunk.transformation_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "split": chunk.split,
                        "text_sha256": chunk.text_sha256,
                        "token_sha256": hashlib.sha256(
                            np.asarray(chunk.token_ids, dtype=np.uint16).tobytes()
                        ).hexdigest(),
                    }
                    logical_digest.update(canonical_json(row).encode("utf-8") + b"\n")
                    observed_chunks += 1
                    observed_real += len(chunk.token_ids)
                    observed_supervised += sum(chunk.stored_mask)
                    decoded_round_trips += 1
                    yield chunk

            with (
                token_path.open("xb") as token_handle,
                mask_path.open("xb") as mask_handle,
                lineage_path.open("xb") as lineage_handle,
            ):
                for record in iter_pack_base_chunks(checked_stream(), split=split):
                    token_handle.write(
                        record.tokens.astype(np.uint16, copy=False).tobytes()
                    )
                    mask_handle.write(
                        record.stored_mask.astype(np.uint8, copy=False).tobytes()
                    )
                    for span in record.spans:
                        item = pending.pop(span.example_id)
                        chunk = item.chunk
                        _write_line(
                            lineage_handle,
                            {
                                "source_id": chunk.source_id,
                                "source_revision": chunk.source_revision,
                                "document_id": chunk.document_id,
                                "document_sha256": chunk.document_sha256,
                                "transformation_id": chunk.transformation_id,
                                "chunk_id": chunk.chunk_id,
                                "chunk_index": chunk.chunk_index,
                                "split": split,
                                "text_sha256": chunk.text_sha256,
                                "packed_record_index": record_count,
                                "start": span.start,
                                "end": span.end,
                                "target_start": span.target_start,
                                "token_count": len(chunk.token_ids),
                                "supervised_target_count": sum(chunk.stored_mask),
                            },
                        )
                    record_count += 1
                if pending:
                    raise ValueError("packed lineage did not consume every chunk")
                for handle in (token_handle, mask_handle, lineage_handle):
                    handle.flush()
                    os.fsync(handle.fileno())

            observed = {
                "chunk_count": observed_chunks,
                "real_token_count": observed_real,
                "supervised_target_count": observed_supervised,
            }
            if observed != expected_table[split]:
                raise ValueError(
                    f"{source_id}/{split} counts do not match the specification"
                )
            paths = {
                "tokens": token_path,
                "mask": mask_path,
                "lineage": lineage_path,
            }
            split_artifacts = {}
            for kind, path in paths.items():
                relative = _relative_artifact_path(source_id, split, kind)
                evidence = _artifact(path)
                artifacts[relative] = evidence
                split_artifacts[kind] = {"path": relative, **evidence}
            source_reports[split] = {
                **observed,
                "packed_record_count": record_count,
                "padding_token_count": record_count * RECORD_WIDTH - observed_real,
                "artifacts": split_artifacts,
            }
            total_chunks += observed_chunks
            total_real += observed_real
            total_supervised += observed_supervised
        split_reports[source_id] = source_reports
        source_totals[source_id] = {
            "chunk_count": sum(
                int(source_reports[split]["chunk_count"]) for split in EXPECTED_SPLITS
            ),
            "real_token_count": sum(
                int(source_reports[split]["real_token_count"])
                for split in EXPECTED_SPLITS
            ),
            "supervised_target_count": sum(
                int(source_reports[split]["supervised_target_count"])
                for split in EXPECTED_SPLITS
            ),
        }

    global_counts = global_validator.finish()
    expected_global = {
        split: sum(
            int(source["expected_counts"][split]["chunk_count"])
            for source in specification["sources"]
        )
        for split in EXPECTED_SPLITS
    }
    if global_counts != expected_global:
        raise ValueError("global split counts do not match source specifications")
    internal_body: dict[str, object] = {
        "schema_id": INTERNAL_MANIFEST_SCHEMA_ID,
        "release_id": specification["release_id"],
        "qualification_scope": specification["qualification_scope"],
        "specification_sha256": specification["specification_sha256"],
        "repository_commit": specification["repository_commit"],
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "record_specification_sha256": SPECIFICATION_SHA256,
        "masking_contract": MASKING_CONTRACT,
        "packing_contract": PACKING_CONTRACT,
        "source_order": list(specification["source_order"]),
        "sources": split_reports,
        "source_totals": {
            source_id: {
                **counts,
                "supervised_share": (
                    f"{counts['supervised_target_count']}/{total_supervised}"
                ),
            }
            for source_id, counts in source_totals.items()
        },
        "input_bindings": {
            "tokenizer": specification["tokenizer"],
            "record_implementation": specification["record_implementation"],
            "builder_implementation": specification["builder_implementation"],
            "source_evidence": {
                str(source["source_id"]): source["evidence"]
                for source in specification["sources"]
            },
            "evaluation_inventories": specification["evaluation_inventories"],
        },
        "global_split_chunk_counts": global_counts,
        "total_chunk_count": total_chunks,
        "total_real_token_count": total_real,
        "total_supervised_target_count": total_supervised,
        "decoded_round_trip_count": decoded_round_trips,
        "logical_order_sha256": logical_digest.hexdigest(),
        "artifacts": dict(sorted(artifacts.items())),
        "publication_authorized": False,
        "training_authorized": False,
    }
    internal_body["manifest_sha256"] = sha256_json(internal_body)
    internal_path = scratch_root / "release.internal.json"
    _write_json(internal_path, internal_body)
    artifacts["release.internal.json"] = _artifact(internal_path)
    external_body: dict[str, object] = {
        "schema_id": EXTERNAL_TEMPLATE_SCHEMA_ID,
        "release_id": specification["release_id"],
        "qualification_scope": specification["qualification_scope"],
        "release_directory": PRODUCTION_RELEASE_PATH,
        "internal_manifest": artifacts["release.internal.json"],
        "specification_sha256": specification["specification_sha256"],
        "repository_commit": specification["repository_commit"],
        "publication_authorized": False,
        "training_authorized": False,
    }
    external_body["template_sha256"] = sha256_json(external_body)
    built = BuiltPass(
        root=scratch_root,
        internal_manifest=internal_body,
        external_template=external_body,
        artifact_evidence=dict(sorted(artifacts.items())),
        pass_sha256=sha256_json(
            {
                "artifact_evidence": dict(sorted(artifacts.items())),
                "external_template_sha256": external_body["template_sha256"],
            }
        ),
    )
    validate_built_pass(built)
    return built


def validate_built_pass(built: BuiltPass) -> None:
    """Validate every serialized output byte and lineage boundary."""

    root = built.root.resolve()
    if not root.is_dir():
        raise ValueError("scratch release directory is missing")
    internal_path = root / "release.internal.json"
    observed_internal = json.loads(internal_path.read_text(encoding="utf-8"))
    if observed_internal != built.internal_manifest:
        raise ValueError("internal manifest bytes do not match the built identity")
    internal_body = dict(observed_internal)
    manifest_hash = internal_body.pop("manifest_sha256", None)
    if sha256_json(internal_body) != manifest_hash:
        raise ValueError("internal manifest identity mismatch")
    if (
        observed_internal.get("publication_authorized") is not False
        or observed_internal.get("training_authorized") is not False
    ):
        raise ValueError("scratch release contains authorization")
    expected_data_artifacts = dict(built.artifact_evidence)
    expected_internal = expected_data_artifacts.pop("release.internal.json", None)
    if observed_internal.get("artifacts") != expected_data_artifacts:
        raise ValueError("internal manifest does not bind every data artifact")
    observed_artifacts: dict[str, dict[str, object]] = {}
    for relative, expected in built.artifact_evidence.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                f"scratch artifact is missing or escapes the root: {relative}"
            )
        evidence = _artifact(path)
        if evidence != expected:
            raise ValueError(f"scratch artifact identity mismatch: {relative}")
        observed_artifacts[relative] = evidence
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != set(built.artifact_evidence):
        raise ValueError("scratch release contains missing or unbound files")

    for source_id, source_report in observed_internal["sources"].items():
        for split in EXPECTED_SPLITS:
            report = source_report[split]
            token_path = root / report["artifacts"]["tokens"]["path"]
            mask_path = root / report["artifacts"]["mask"]["path"]
            lineage_path = root / report["artifacts"]["lineage"]["path"]
            tokens = np.fromfile(token_path, dtype=np.uint16)
            masks = np.fromfile(mask_path, dtype=np.uint8)
            if tokens.size != masks.size or tokens.size % RECORD_WIDTH:
                raise ValueError(f"{source_id}/{split} token/mask layout mismatch")
            token_rows = tokens.reshape(-1, RECORD_WIDTH)
            mask_rows = masks.reshape(-1, RECORD_WIDTH)
            if len(token_rows) != report["packed_record_count"]:
                raise ValueError(f"{source_id}/{split} record count mismatch")
            if not np.isin(mask_rows, (0, 1)).all():
                raise ValueError(f"{source_id}/{split} mask is not binary")
            rows = [
                json.loads(line)
                for line in lineage_path.read_text(encoding="utf-8").splitlines()
            ]
            if len(rows) != report["chunk_count"]:
                raise ValueError(f"{source_id}/{split} lineage count mismatch")
            by_record: dict[int, list[Mapping[str, object]]] = {}
            for row in rows:
                if row["source_id"] != source_id or row["split"] != split:
                    raise ValueError("lineage source/split mismatch")
                by_record.setdefault(int(row["packed_record_index"]), []).append(row)
            if set(by_record) != set(range(len(token_rows))):
                raise ValueError(f"{source_id}/{split} record lineage is incomplete")
            counted_real = 0
            counted_supervised = 0
            for record_index, spans in by_record.items():
                spans.sort(key=lambda row: int(row["start"]))
                expected_start = 0
                for row in spans:
                    start = int(row["start"])
                    end = int(row["end"])
                    target_start = int(row["target_start"])
                    if (
                        start != expected_start
                        or target_start != start + 1
                        or end <= target_start
                    ):
                        raise ValueError("lineage contains an invalid packed span")
                    if int(mask_rows[record_index, start]) != 0:
                        raise ValueError("chunk boundary target is supervised")
                    if not np.all(mask_rows[record_index, target_start:end] == 1):
                        raise ValueError(
                            "full-loss chunk targets are not fully supervised"
                        )
                    if int(token_rows[record_index, end - 1]) != EOS_TOKEN_ID:
                        raise ValueError("serialized chunk lacks terminal EOS")
                    counted_real += end - start
                    counted_supervised += end - target_start
                    expected_start = end
                if np.any(token_rows[record_index, expected_start:] != PAD_TOKEN_ID):
                    raise ValueError("serialized PAD tail contains content")
                if np.any(mask_rows[record_index, expected_start:] != 0):
                    raise ValueError("serialized PAD tail is supervised")
            if counted_real != report["real_token_count"]:
                raise ValueError("serialized real-token count mismatch")
            if counted_supervised != report["supervised_target_count"]:
                raise ValueError("serialized supervised-target count mismatch")
    if observed_artifacts != built.artifact_evidence:
        raise ValueError("scratch artifact evidence mismatch")
    external_body = dict(built.external_template)
    template_hash = external_body.pop("template_sha256", None)
    if sha256_json(external_body) != template_hash:
        raise ValueError("external manifest template identity mismatch")
    if built.external_template.get("internal_manifest") != expected_internal:
        raise ValueError("external template does not bind the internal manifest")
    expected_pass = sha256_json(
        {
            "artifact_evidence": built.artifact_evidence,
            "external_template_sha256": built.external_template["template_sha256"],
        }
    )
    if built.pass_sha256 != expected_pass:
        raise ValueError("scratch pass identity mismatch")


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    isjunction = getattr(os.path, "isjunction", None)
    return bool(isjunction and isjunction(path))


def _validate_scratch_target(path: Path, parent: Path) -> None:
    absolute_parent = parent.resolve()
    candidate = path.absolute()
    if candidate.parent.resolve() != absolute_parent:
        raise ValueError("scratch pass must be a direct child of the scratch parent")
    current = candidate.parent
    while True:
        if current.exists() and _is_link_or_junction(current):
            raise ValueError(f"scratch path traverses a link or junction: {current}")
        if current == absolute_parent:
            break
        current = current.parent
    if candidate.exists():
        raise FileExistsError(f"scratch pass already exists: {candidate}")


def _estimated_scratch_bytes(specification: Mapping[str, object]) -> int:
    records = 0
    chunks = 0
    for source in specification["sources"]:
        for split in EXPECTED_SPLITS:
            counts = source["expected_counts"][split]
            chunks += int(counts["chunk_count"])
            records += int(counts["chunk_count"])
    return records * RECORD_WIDTH * 3 + chunks * 1024 + 1_048_576


def qualify_base_release(
    *,
    specification: Mapping[str, object],
    repository_root: Path,
    runtime_commit: str,
    tokenizer: RoundTripTokenizer,
    stream_factory: StreamFactory,
    scratch_parent: Path,
    free_bytes: Callable[[Path], int] | None = None,
    mutation_hook: Callable[[Path], None] | None = None,
) -> dict[str, object]:
    """Run two scratch builds and return non-authorizing qualification evidence."""

    validate_release_specification_files(
        specification, repository_root, runtime_commit=runtime_commit
    )
    raw_parent = scratch_parent.absolute()
    if not raw_parent.is_dir() or _is_link_or_junction(raw_parent):
        raise ValueError("scratch parent must be an existing canonical directory")
    parent = raw_parent.resolve()
    pass_paths = [
        parent / ".vasu-140m-base-qualification-pass-1",
        parent / ".vasu-140m-base-qualification-pass-2",
    ]
    for path in pass_paths:
        _validate_scratch_target(path, parent)
    required = max(
        int(specification["minimum_free_bytes"]),
        _estimated_scratch_bytes(specification),
    )
    disk = free_bytes or (lambda path: shutil.disk_usage(path).free)
    pass_evidence: list[dict[str, object]] = []
    first_identity: dict[str, object] | None = None
    first_artifacts: Mapping[str, Mapping[str, object]] | None = None
    try:
        for pass_index, path in enumerate(pass_paths, start=1):
            if disk(parent) < required:
                raise OSError("insufficient free disk for base-release qualification")
            built = _build_pass(
                specification=specification,
                tokenizer=tokenizer,
                stream_factory=stream_factory,
                scratch_root=path,
            )
            validate_built_pass(built)
            if pass_index == 2 and mutation_hook is not None:
                mutation_hook(path)
                validate_built_pass(built)
            identity = {
                "pass_sha256": built.pass_sha256,
                "internal_manifest_sha256": built.internal_manifest["manifest_sha256"],
                "external_template_sha256": built.external_template["template_sha256"],
            }
            if first_identity is None:
                first_identity = identity
                first_artifacts = built.artifact_evidence
            elif identity != first_identity:
                raise ValueError("independent scratch builds are not byte-identical")
            pass_evidence.append(identity)
            shutil.rmtree(path)
        assert first_identity is not None and first_artifacts is not None
        report: dict[str, object] = {
            "schema_id": QUALIFICATION_SCHEMA_ID,
            "release_id": specification["release_id"],
            "qualification_scope": specification["qualification_scope"],
            "specification_sha256": specification["specification_sha256"],
            "repository_commit": specification["repository_commit"],
            "builder_implementation_sha256": specification["builder_implementation"][
                "sha256"
            ],
            "record_implementation_sha256": specification["record_implementation"][
                "sha256"
            ],
            "pass_count": 2,
            "passes": pass_evidence,
            "expected_production_artifacts": first_artifacts,
            "expected_internal_manifest_sha256": first_identity[
                "internal_manifest_sha256"
            ],
            "expected_external_template_sha256": first_identity[
                "external_template_sha256"
            ],
            "round_trip_complete": True,
            "post_validation_mutation_check": True,
            "scratch_cleaned": True,
            "production_input_evidence_validated": (
                specification["qualification_scope"] == "production"
            ),
            "production_release_created": False,
            "publication_authorized": False,
            "training_authorized": False,
        }
        report["qualification_sha256"] = sha256_json(report)
        validate_qualification_report(report)
        return report
    except Exception:
        # Failure evidence is intentionally preserved for review.
        raise


def validate_qualification_report(report: Mapping[str, object]) -> None:
    """Validate deterministic, explicitly non-authorizing qualification evidence."""

    _exact(
        report,
        {
            "schema_id",
            "release_id",
            "qualification_scope",
            "specification_sha256",
            "repository_commit",
            "builder_implementation_sha256",
            "record_implementation_sha256",
            "pass_count",
            "passes",
            "expected_production_artifacts",
            "expected_internal_manifest_sha256",
            "expected_external_template_sha256",
            "round_trip_complete",
            "post_validation_mutation_check",
            "scratch_cleaned",
            "production_input_evidence_validated",
            "production_release_created",
            "publication_authorized",
            "training_authorized",
            "qualification_sha256",
        },
        "base release qualification report",
    )
    if report["schema_id"] != QUALIFICATION_SCHEMA_ID or report["pass_count"] != 2:
        raise ValueError("base release qualification schema/pass count mismatch")
    _string(report["release_id"], "release_id")
    if report["qualification_scope"] not in {"fixture", "production"}:
        raise ValueError("qualification scope is invalid")
    if report["production_input_evidence_validated"] is not (
        report["qualification_scope"] == "production"
    ):
        raise ValueError("qualification scope/evidence status mismatch")
    for field in (
        "specification_sha256",
        "builder_implementation_sha256",
        "record_implementation_sha256",
        "expected_internal_manifest_sha256",
        "expected_external_template_sha256",
    ):
        _sha(report[field], field)
    _commit(report["repository_commit"], "repository_commit")
    passes = report["passes"]
    if not isinstance(passes, list) or len(passes) != 2 or passes[0] != passes[1]:
        raise ValueError("qualification passes are incomplete or non-identical")
    pass_fields = {
        "pass_sha256",
        "internal_manifest_sha256",
        "external_template_sha256",
    }
    expected_artifacts = _mapping(
        report["expected_production_artifacts"], "expected_production_artifacts"
    )
    if not expected_artifacts:
        raise ValueError("qualification expected artifact evidence is empty")
    for relative, raw_evidence in expected_artifacts.items():
        _safe_path(relative, "qualification artifact path")
        evidence = _mapping(raw_evidence, f"qualification artifact {relative}")
        _exact(evidence, {"bytes", "sha256"}, f"qualification artifact {relative}")
        _positive_int(evidence["bytes"], f"qualification artifact {relative}.bytes")
        _sha(evidence["sha256"], f"qualification artifact {relative}.sha256")
    for index, raw in enumerate(passes):
        item = _mapping(raw, f"qualification pass {index}")
        _exact(item, pass_fields, f"qualification pass {index}")
        _sha(item["pass_sha256"], f"qualification pass {index}.pass_sha256")
        _sha(
            item["internal_manifest_sha256"],
            f"qualification pass {index}.internal_manifest_sha256",
        )
        _sha(
            item["external_template_sha256"],
            f"qualification pass {index}.external_template_sha256",
        )
        expected_pass_sha = sha256_json(
            {
                "artifact_evidence": expected_artifacts,
                "external_template_sha256": item["external_template_sha256"],
            }
        )
        if item["pass_sha256"] != expected_pass_sha:
            raise ValueError("qualification pass identity mismatch")
    if report["expected_internal_manifest_sha256"] != passes[0].get(
        "internal_manifest_sha256"
    ):
        raise ValueError("qualification internal manifest identity mismatch")
    if report["expected_external_template_sha256"] != passes[0].get(
        "external_template_sha256"
    ):
        raise ValueError("qualification external template identity mismatch")
    for field in (
        "round_trip_complete",
        "post_validation_mutation_check",
        "scratch_cleaned",
    ):
        if report[field] is not True:
            raise ValueError(f"qualification {field} must be true")
    for field in (
        "production_release_created",
        "publication_authorized",
        "training_authorized",
    ):
        if report[field] is not False:
            raise ValueError(f"qualification {field} must be false")
    reported = _sha(report["qualification_sha256"], "qualification_sha256")
    body = dict(report)
    del body["qualification_sha256"]
    if sha256_json(body) != reported:
        raise ValueError("base release qualification identity mismatch")
