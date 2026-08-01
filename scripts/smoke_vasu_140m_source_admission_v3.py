"""Qualify VASU-140M source-admission v3 against both candidate registries."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.sources import load_source_records  # noqa: E402
from vasu.data.vasu_140m_source_admission import (  # noqa: E402
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    SCHEMA_ID,
    TOKENIZER_SHA256,
    package_identity,
    validate_admission_package_files,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    *,
    registry_relative: str,
    source_id: str,
    shard_evidence_relative: str,
    unresolved_items: list[str],
    obligations: list[str],
    stable_id_fields: list[str],
    transformation_id: str,
) -> dict[str, object]:
    registry_path = REPOSITORY_ROOT / registry_relative
    records = [
        record
        for record in load_source_records(registry_path)
        if record.source_id == source_id
    ]
    if len(records) != 1:
        raise ValueError(f"candidate source does not resolve exactly once: {source_id}")
    record = records[0]
    inventory_relative = "evaluation/benchmarks/ultrachat_promotion_v1.json"
    inventory_path = REPOSITORY_ROOT / inventory_relative
    shard_path = REPOSITORY_ROOT / shard_evidence_relative
    package: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "package_id": f"vasu-140m-{source_id}-admission-v3-candidate",
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "source_record": {
            "path": registry_relative,
            "sha256": _sha256(registry_path),
            "source_id": source_id,
            "registry_approval_status": record.approval_status,
        },
        "legal_evidence": {
            "license_name": record.license_name,
            "license_url": record.license_url,
            "terms_url": (
                "https://commoncrawl.org/terms-of-use"
                if source_id == "fineweb_edu_extension_2025_26"
                else "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use"
            ),
            "terms_revision": "primary-source review 2026-08-01",
            "commercial_use_allowed": record.commercial_use_allowed,
            "attribution_required": record.attribution_required,
            "redistribution_allowed": record.redistribution_allowed,
            "gated_access": record.gated_access,
            "requires_authentication": record.requires_authentication,
            "obligations": obligations,
            "unresolved_items": unresolved_items,
        },
        "acquisition": {
            "immutable_revision": record.pinned_revision,
            "access_method": record.access_method,
            "shard_inventory_sha256": _sha256(shard_path),
            "raw_hash_algorithm": "sha256",
            "authorized": False,
        },
        "document_lineage": {
            "stable_id_fields": stable_id_fields,
            "revision_field": "pinned_revision",
            "shard_field": "source_shard",
            "transformation_id": transformation_id,
        },
        "quality_policy": {
            "normalization_version": "unicode_nfc_whitespace_v1",
            "filter_version": f"vasu_140m_{source_id}_filter_candidate_v1",
            "rejection_reasons": [
                "empty_or_malformed_text",
                "unresolved_rights_or_policy",
                "evaluation_contamination",
                "exact_or_near_duplicate",
            ],
        },
        "evaluation_isolation": {
            "inventories": [
                {
                    "inventory_id": "legacy-placeholder-not-production",
                    "dimension": "factuality",
                    "split": "development",
                    "path": inventory_relative,
                    "sha256": _sha256(inventory_path),
                }
            ],
            "likelihood_policy": {
                "created_after_acquisition": True,
                "document_level_isolation": True,
                "train_exclusion_required": True,
                "manifest_binding_required": True,
            },
            "exact_method": "NFC exact prompt/answer and >=8-word fragment hashes",
            "ngram_words": 8,
            "scan_before_split": True,
        },
        "deduplication": {
            "normalization_version": "unicode_nfc_whitespace_v1",
            "exact_method": "sha256",
            "near_method": "deterministic_minhash_lsh_word_5gram_v1",
            "cross_source_indexes": [
                "configs/data/deduplication/fineweb_index.json",
                "configs/data/deduplication/fineweb_extension_recovery_production.json",
            ],
            "before_split": True,
        },
        "decision": {
            "state": "blocked",
            "reviewed_by": "",
            "reviewed_at": "",
            "notes": (
                "Candidate evidence only. Production evaluation inventories and "
                "source-specific policy review remain unresolved."
            ),
        },
        "training_authorized": False,
    }
    package["package_sha256"] = package_identity(package)
    validate_admission_package_files(package, REPOSITORY_ROOT)
    return package


def build_report() -> dict[str, object]:
    fineweb = _candidate(
        registry_relative="configs/data/sources/fineweb_edu.json",
        source_id="fineweb_edu_extension_2025_26",
        shard_evidence_relative=(
            "data/manifests/pretrain/fineweb_extension_recovery_production.json"
        ),
        unresolved_items=[
            "ODC-By covers the database but not independent rights in web contents",
            "Common Crawl terms and third-party content-rights disposition require review",
            "all 12 production-candidate evaluation inventories are not yet frozen",
        ],
        obligations=[
            "attribute FineWeb-Edu and preserve ODC-By notices",
            "preserve source URL, document ID, crawl dump, and recovery lineage",
            "respect Common Crawl terms and third-party rights",
        ],
        stable_id_fields=["id", "url", "dump", "file_path"],
        transformation_id="fineweb_edu_extension_recovery_to_vasu_140m_v1",
    )
    wikipedia = _candidate(
        registry_relative="configs/data/sources/wikimedia.json",
        source_id="wikipedia_en_20231101_planned",
        shard_evidence_relative="configs/data/locks/wikimedia_factual_pilot_v1.json",
        unresolved_items=[
            "imported or fair-use material exclusion policy requires review",
            "attribution and ShareAlike publication mechanics require review",
            "all 12 production-candidate evaluation inventories are not yet frozen",
        ],
        obligations=[
            "preserve article ID, title, URL, revision, and attribution",
            "provide CC-BY-SA and GFDL notices and change notices",
            "apply ShareAlike where an adaptation triggers it",
        ],
        stable_id_fields=["id", "url", "title"],
        transformation_id="wikipedia_20231101_en_to_vasu_140m_v1",
    )
    return {
        "schema_id": "vasu_140m_source_admission_v3_qualification_v1",
        "contract_schema_id": SCHEMA_ID,
        "candidate_package_sha256": {
            "fineweb_edu_extension_2025_26": fineweb["package_sha256"],
            "wikipedia_en_20231101_planned": wikipedia["package_sha256"],
        },
        "candidate_states": {
            "fineweb_edu_extension_2025_26": fineweb["decision"]["state"],
            "wikipedia_en_20231101_planned": wikipedia["decision"]["state"],
        },
        "complete_production_prompt_inventory_matrix_bound": False,
        "source_acquisition_authorized": False,
        "source_bytes_downloaded": False,
        "production_release_created": False,
        "training_authorized": False,
    }


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
