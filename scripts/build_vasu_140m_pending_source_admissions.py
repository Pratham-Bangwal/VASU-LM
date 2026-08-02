"""Build immutable pending admission packages for the two VASU-140M sources."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vasu.data.sources import load_source_records  # noqa: E402
from vasu.data.vasu_140m_source_admission import (  # noqa: E402
    FAMILY_ID,
    MODEL_CONFIG_SHA256,
    SCHEMA_ID,
    TOKENIZER_SHA256,
    package_identity,
    validate_admission_package_files,
)


OUTPUT = ROOT / "configs/data/admissions"
SUITE = ROOT / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
DIMENSIONS = ("factuality", "arithmetic", "repetition", "robustness", "manual_review")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventories() -> list[dict[str, object]]:
    values = []
    for dimension in DIMENSIONS:
        path = SUITE / dimension / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        values.append(
            {
                "inventory_id": manifest["inventory_id"],
                "dimension": dimension,
                "split": "development",
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return values


def source_record(path: Path, source_id: str):
    return next(record for record in load_source_records(path) if record.source_id == source_id)


def build_package(*, registry: str, source_id: str, terms_url: str, terms_revision: str, obligations: list[str], unresolved: list[str], stable_ids: list[str], revision_field: str, shard_field: str, filter_version: str, rejection_reasons: list[str], cross_indexes: list[str]) -> dict[str, object]:
    registry_path = ROOT / registry
    record = source_record(registry_path, source_id)
    package: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "package_id": f"vasu-140m-{source_id}-pending-v1",
        "family_id": FAMILY_ID,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "source_record": {
            "path": registry,
            "sha256": sha256_file(registry_path),
            "source_id": source_id,
            "registry_approval_status": record.approval_status,
        },
        "legal_evidence": {
            "license_name": record.license_name,
            "license_url": record.license_url,
            "terms_url": terms_url,
            "terms_revision": terms_revision,
            "commercial_use_allowed": record.commercial_use_allowed,
            "attribution_required": record.attribution_required,
            "redistribution_allowed": record.redistribution_allowed,
            "gated_access": record.gated_access,
            "requires_authentication": record.requires_authentication,
            "obligations": obligations,
            "unresolved_items": unresolved,
        },
        "acquisition": {
            "immutable_revision": record.pinned_revision,
            "access_method": record.access_method,
            "shard_inventory_sha256": hashlib.sha256(f"pending:{source_id}:shard-inventory".encode()).hexdigest(),
            "raw_hash_algorithm": "sha256",
            "authorized": False,
        },
        "document_lineage": {
            "stable_id_fields": stable_ids,
            "revision_field": revision_field,
            "shard_field": shard_field,
            "transformation_id": "vasu_140m_base_source_filter_v1",
        },
        "quality_policy": {
            "normalization_version": "unicode_nfc_whitespace_v1",
            "filter_version": filter_version,
            "rejection_reasons": rejection_reasons,
        },
        "evaluation_isolation": {
            "inventories": inventories(),
            "likelihood_policy": {
                "created_after_acquisition": True,
                "document_level_isolation": True,
                "train_exclusion_required": True,
                "manifest_binding_required": True,
            },
            "exact_method": "sha256_nfc_casefold_whitespace_v1",
            "ngram_words": 8,
            "scan_before_split": True,
        },
        "deduplication": {
            "normalization_version": "unicode_nfc_whitespace_v1",
            "exact_method": "sha256",
            "near_method": "minhash_lsh_word_5gram_v1",
            "cross_source_indexes": cross_indexes,
            "before_split": True,
        },
        "decision": {
            "state": "pending",
            "reviewed_by": "",
            "reviewed_at": "",
            "notes": "Pending acquisition identity, production inventory matrix, contamination scan, and source-specific review.",
        },
        "training_authorized": False,
        "package_sha256": "0" * 64,
    }
    package["package_sha256"] = package_identity(package)
    return package


def main() -> None:
    packages = {
        "fineweb_edu_extension_2025_26.pending.json": build_package(
            registry="configs/data/sources/fineweb_edu.json",
            source_id="fineweb_edu_extension_2025_26",
            terms_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            terms_revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            obligations=["retain ODC-By attribution", "preserve document provenance", "review underlying web-document rights before redistribution"],
            unresolved=["exact immutable raw shard inventory is not yet acquired", "all ten production prompt inventories are not yet available", "pre-split contamination and cross-source deduplication are not yet executed"],
            stable_ids=["source_document_id", "text_sha256"], revision_field="dump", shard_field="shard",
            filter_version="fineweb_edu_extension_v1",
            rejection_reasons=["missing source ID", "invalid text", "evaluation contamination", "exact or near duplicate"],
            cross_indexes=["fineweb_original_plus_extension_v1", "wikipedia_en_20231101_v1"],
        ),
        "wikipedia_en_20231101.pending.json": build_package(
            registry="configs/data/sources/wikimedia.json",
            source_id="wikipedia_en_20231101_planned",
            terms_url="https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
            terms_revision="20231101.en snapshot policy binding",
            obligations=["retain article-level attribution and source URL", "retain CC-BY-SA and GFDL notices", "record modifications", "apply ShareAlike where required"],
            unresolved=["exact immutable Parquet shard inventory is not yet acquired", "all ten production prompt inventories are not yet available", "pre-split contamination and cross-source deduplication are not yet executed"],
            stable_ids=["id", "url", "title"], revision_field="snapshot_revision", shard_field="parquet_shard",
            filter_version="wikipedia_en_20231101_v1",
            rejection_reasons=["non-article namespace", "redirect or disambiguation", "markup or boilerplate", "invalid Unicode", "evaluation contamination", "exact or near duplicate"],
            cross_indexes=["fineweb_original_plus_extension_v1", "wikipedia_en_20231101_v1"],
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, package in packages.items():
        path = OUTPUT / name
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(package, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        validate_admission_package_files(package, ROOT)
    print(json.dumps({name: package["package_sha256"] for name, package in packages.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
