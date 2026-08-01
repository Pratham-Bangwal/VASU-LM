"""Strict metadata contract for VASU-140M base-source admission packages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from vasu.data.sources import load_source_records


SCHEMA_ID = "vasu_140m_base_source_admission_v3"
LEGACY_SCHEMA_ID = "vasu_140m_base_source_admission_v1"
PREVIOUS_SCHEMA_ID = "vasu_140m_base_source_admission_v2"
FAMILY_ID = "vasu_140m_v1"
MODEL_CONFIG_SHA256 = (
    "29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059"
)
TOKENIZER_SHA256 = (
    "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
)
EVALUATION_DIMENSIONS = frozenset(
    {
        "likelihood",
        "factuality",
        "arithmetic",
        "repetition",
        "robustness",
        "manual_review",
    }
)
EVALUATION_SPLITS = frozenset({"development", "held_out"})
PROMPT_EVALUATION_DIMENSIONS = EVALUATION_DIMENSIONS - {"likelihood"}
STATES = frozenset({"pending", "blocked", "rejected", "approved"})
SHA256_HEX_LENGTH = 64


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def package_identity(package: Mapping[str, object]) -> str:
    body = dict(package)
    body.pop("package_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{label} fields mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != SHA256_HEX_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _strings(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} must be a list of non-empty strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def _url(value: object, label: str) -> str:
    text = _string(value, label)
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return text


def _false(value: object, label: str) -> None:
    if value is not False:
        raise ValueError(f"{label} must be false")


def _timezone_timestamp(value: object, label: str) -> str:
    text = _string(value, label)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return text


def _safe_repository_path(value: object, label: str) -> str:
    text = _string(value, label).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a repository-relative path")
    return text


def validate_admission_package(package: Mapping[str, object]) -> None:
    """Fail closed unless a source-admission package satisfies the v2 contract."""

    _exact_keys(
        package,
        {
            "schema_id", "package_id", "family_id", "model_config_sha256",
            "tokenizer_sha256", "source_record", "legal_evidence", "acquisition",
            "document_lineage", "quality_policy", "evaluation_isolation",
            "deduplication", "decision", "training_authorized", "package_sha256",
        },
        "admission package",
    )
    if package["schema_id"] != SCHEMA_ID or package["family_id"] != FAMILY_ID:
        raise ValueError("admission package family/schema identity mismatch")
    _string(package["package_id"], "package_id")
    if _sha256(package["model_config_sha256"], "model_config_sha256") != MODEL_CONFIG_SHA256:
        raise ValueError("model configuration identity mismatch")
    if _sha256(package["tokenizer_sha256"], "tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity mismatch")
    _false(package["training_authorized"], "training_authorized")

    source = _mapping(package["source_record"], "source_record")
    _exact_keys(
        source,
        {"path", "sha256", "source_id", "registry_approval_status"},
        "source_record",
    )
    source_path = _safe_repository_path(source["path"], "source_record.path")
    if not source_path.startswith("configs/data/sources/") or not source_path.endswith(".json"):
        raise ValueError("source_record.path must be a source-registry JSON path")
    _sha256(source["sha256"], "source_record.sha256")
    _string(source["source_id"], "source_record.source_id")
    if source["registry_approval_status"] not in STATES:
        raise ValueError("source_record.registry_approval_status is unsupported")

    legal = _mapping(package["legal_evidence"], "legal_evidence")
    _exact_keys(
        legal,
        {
            "license_name",
            "license_url",
            "terms_url",
            "terms_revision",
            "commercial_use_allowed",
            "attribution_required",
            "redistribution_allowed",
            "gated_access",
            "requires_authentication",
            "obligations",
            "unresolved_items",
        },
        "legal_evidence",
    )
    _string(legal["license_name"], "legal_evidence.license_name")
    _url(legal["license_url"], "legal_evidence.license_url")
    _url(legal["terms_url"], "legal_evidence.terms_url")
    _string(legal["terms_revision"], "legal_evidence.terms_revision")
    for field in (
        "commercial_use_allowed",
        "attribution_required",
        "redistribution_allowed",
        "gated_access",
        "requires_authentication",
    ):
        if not isinstance(legal[field], bool):
            raise ValueError(f"legal_evidence.{field} must be boolean")
    _strings(legal["obligations"], "legal_evidence.obligations")
    unresolved = _strings(legal["unresolved_items"], "legal_evidence.unresolved_items", allow_empty=True)

    acquisition = _mapping(package["acquisition"], "acquisition")
    _exact_keys(
        acquisition,
        {"immutable_revision", "access_method", "shard_inventory_sha256", "raw_hash_algorithm", "authorized"},
        "acquisition",
    )
    _string(acquisition["immutable_revision"], "acquisition.immutable_revision")
    _string(acquisition["access_method"], "acquisition.access_method")
    _sha256(acquisition["shard_inventory_sha256"], "acquisition.shard_inventory_sha256")
    if acquisition["raw_hash_algorithm"] != "sha256":
        raise ValueError("acquisition.raw_hash_algorithm must be sha256")
    _false(acquisition["authorized"], "acquisition.authorized")

    lineage = _mapping(package["document_lineage"], "document_lineage")
    _exact_keys(lineage, {"stable_id_fields", "revision_field", "shard_field", "transformation_id"}, "document_lineage")
    _strings(lineage["stable_id_fields"], "document_lineage.stable_id_fields")
    _string(lineage["revision_field"], "document_lineage.revision_field")
    _string(lineage["shard_field"], "document_lineage.shard_field")
    _string(lineage["transformation_id"], "document_lineage.transformation_id")

    quality = _mapping(package["quality_policy"], "quality_policy")
    _exact_keys(quality, {"normalization_version", "filter_version", "rejection_reasons"}, "quality_policy")
    _string(quality["normalization_version"], "quality_policy.normalization_version")
    _string(quality["filter_version"], "quality_policy.filter_version")
    _strings(quality["rejection_reasons"], "quality_policy.rejection_reasons")

    isolation = _mapping(package["evaluation_isolation"], "evaluation_isolation")
    _exact_keys(
        isolation,
        {
            "inventories",
            "likelihood_policy",
            "exact_method",
            "ngram_words",
            "scan_before_split",
        },
        "evaluation_isolation",
    )
    inventories = isolation["inventories"]
    if not isinstance(inventories, list) or not inventories:
        raise ValueError("evaluation_isolation.inventories must be non-empty")
    seen_paths: set[str] = set()
    for index, item in enumerate(inventories):
        inventory = _mapping(item, f"evaluation inventory {index}")
        _exact_keys(
            inventory,
            {"inventory_id", "dimension", "split", "path", "sha256"},
            f"evaluation inventory {index}",
        )
        _string(inventory["inventory_id"], f"evaluation inventory {index}.inventory_id")
        if inventory["dimension"] not in PROMPT_EVALUATION_DIMENSIONS:
            raise ValueError(
                f"evaluation inventory {index}.dimension is not a prompt dimension"
            )
        if inventory["split"] not in EVALUATION_SPLITS:
            raise ValueError(f"evaluation inventory {index}.split is unsupported")
        path = _safe_repository_path(inventory["path"], f"evaluation inventory {index}.path")
        if path in seen_paths:
            raise ValueError("evaluation inventory paths must be unique")
        seen_paths.add(path)
        _sha256(inventory["sha256"], f"evaluation inventory {index}.sha256")
    _string(isolation["exact_method"], "evaluation_isolation.exact_method")
    if not isinstance(isolation["ngram_words"], int) or isolation["ngram_words"] < 5:
        raise ValueError("evaluation_isolation.ngram_words must be an integer >= 5")
    if isolation["scan_before_split"] is not True:
        raise ValueError("evaluation isolation must run before split assignment")
    likelihood_policy = _mapping(
        isolation["likelihood_policy"],
        "evaluation_isolation.likelihood_policy",
    )
    _exact_keys(
        likelihood_policy,
        {
            "created_after_acquisition",
            "document_level_isolation",
            "train_exclusion_required",
            "manifest_binding_required",
        },
        "evaluation_isolation.likelihood_policy",
    )
    if any(value is not True for value in likelihood_policy.values()):
        raise ValueError(
            "likelihood evaluation must be document-isolated after acquisition"
        )

    dedup = _mapping(package["deduplication"], "deduplication")
    _exact_keys(dedup, {"normalization_version", "exact_method", "near_method", "cross_source_indexes", "before_split"}, "deduplication")
    _string(dedup["normalization_version"], "deduplication.normalization_version")
    if dedup["exact_method"] != "sha256":
        raise ValueError("deduplication.exact_method must be sha256")
    _string(dedup["near_method"], "deduplication.near_method")
    _strings(dedup["cross_source_indexes"], "deduplication.cross_source_indexes")
    if dedup["before_split"] is not True:
        raise ValueError("deduplication must run before split assignment")

    decision = _mapping(package["decision"], "decision")
    _exact_keys(decision, {"state", "reviewed_by", "reviewed_at", "notes"}, "decision")
    state = decision["state"]
    if state not in STATES:
        raise ValueError("decision state is unsupported")
    _string(decision["notes"], "decision.notes")
    if state == "approved":
        if source["registry_approval_status"] != "approved":
            raise ValueError(
                "VASU-140M approval requires an approved generic source record"
            )
        _string(decision["reviewed_by"], "decision.reviewed_by")
        _timezone_timestamp(decision["reviewed_at"], "decision.reviewed_at")
        if unresolved:
            raise ValueError("approved admission package has unresolved legal items")
        inventory_pairs = {
            (inventory["dimension"], inventory["split"])
            for inventory in inventories
        }
        expected_pairs = {
            (dimension, split)
            for dimension in PROMPT_EVALUATION_DIMENSIONS
            for split in EVALUATION_SPLITS
        }
        if inventory_pairs != expected_pairs or len(inventories) != len(expected_pairs):
            raise ValueError(
                "approved admission package must bind all 10 prompt inventories"
            )
        inventory_ids = [inventory["inventory_id"] for inventory in inventories]
        if len(inventory_ids) != len(set(inventory_ids)):
            raise ValueError("approved evaluation inventory IDs must be unique")
    elif not isinstance(decision["reviewed_by"], str) or not isinstance(decision["reviewed_at"], str):
        raise ValueError("decision reviewer fields must be strings")

    reported = _sha256(package["package_sha256"], "package_sha256")
    if reported != package_identity(package):
        raise ValueError("admission package identity mismatch")


def validate_admission_package_files(
    package: Mapping[str, object], repository_root: Path
) -> None:
    """Verify hash-bound registry and evaluation files inside a repository."""

    validate_admission_package(package)
    root = repository_root.resolve()
    source = _mapping(package["source_record"], "source_record")
    source_path = (root / _safe_repository_path(source["path"], "source_record.path")).resolve()
    if not source_path.is_relative_to(root):
        raise ValueError("source_record.path escapes the repository")
    if not source_path.is_file():
        raise ValueError("source_record.path does not exist")
    observed_source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if observed_source_hash != source["sha256"]:
        raise ValueError("source registry file identity mismatch")
    matches = [
        record
        for record in load_source_records(source_path)
        if record.source_id == source["source_id"]
    ]
    if len(matches) != 1:
        raise ValueError("source record ID does not resolve exactly once")
    record = matches[0]
    if record.approval_status != source["registry_approval_status"]:
        raise ValueError("source record approval state mismatch")
    legal = _mapping(package["legal_evidence"], "legal_evidence")
    registry_legal = {
        "license_name": record.license_name,
        "license_url": record.license_url,
        "commercial_use_allowed": record.commercial_use_allowed,
        "attribution_required": record.attribution_required,
        "redistribution_allowed": record.redistribution_allowed,
        "gated_access": record.gated_access,
        "requires_authentication": record.requires_authentication,
    }
    for field, expected in registry_legal.items():
        if legal[field] != expected:
            raise ValueError(
                f"legal_evidence.{field} does not match the source registry"
            )
    acquisition = _mapping(package["acquisition"], "acquisition")
    if acquisition["immutable_revision"] != record.pinned_revision:
        raise ValueError(
            "acquisition.immutable_revision does not match the source registry"
        )
    if acquisition["access_method"] != record.access_method:
        raise ValueError(
            "acquisition.access_method does not match the source registry"
        )

    isolation = _mapping(package["evaluation_isolation"], "evaluation_isolation")
    inventories = isolation["inventories"]
    assert isinstance(inventories, list)
    for index, item in enumerate(inventories):
        inventory = _mapping(item, f"evaluation inventory {index}")
        inventory_path = (
            root
            / _safe_repository_path(
                inventory["path"], f"evaluation inventory {index}.path"
            )
        ).resolve()
        if not inventory_path.is_relative_to(root):
            raise ValueError(f"evaluation inventory {index} escapes the repository")
        if not inventory_path.is_file():
            raise ValueError(f"evaluation inventory {index} does not exist")
        observed = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        if observed != inventory["sha256"]:
            raise ValueError(f"evaluation inventory {index} identity mismatch")
        if package["decision"]["state"] == "approved":
            try:
                manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"evaluation inventory {index} is not a JSON manifest"
                ) from error
            if not isinstance(manifest, Mapping):
                raise ValueError(
                    f"evaluation inventory {index} must contain a JSON object"
                )
            for field in ("inventory_id", "dimension", "split"):
                if manifest.get(field) != inventory[field]:
                    raise ValueError(
                        f"evaluation inventory {index} {field} mismatch"
                    )
            if manifest.get("fixture_only") is not False:
                raise ValueError(
                    f"evaluation inventory {index} is not production-candidate evidence"
                )


def load_admission_package(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("admission package must contain a JSON object")
    validate_admission_package(payload)
    return payload
