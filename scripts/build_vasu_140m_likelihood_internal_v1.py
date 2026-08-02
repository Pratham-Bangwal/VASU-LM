"""Build the non-independent VASU-140M likelihood qualification inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_base_v2 import sha256_file  # noqa: E402
from evaluation.framework.vasu_140m_likelihood_inventory import (  # noqa: E402
    ITEMS_PER_SPLIT,
    build_inventory,
    build_records,
    iter_fineweb_documents,
    iter_wikimedia_documents,
    reserve_documents,
)


SUITE_ID = "vasu-140m-base-evaluation-v2-likelihood-self-curated-v1"
OUTPUT_ROOT = "evaluation/fixtures/vasu_140m_likelihood_self_curated_v1"
FINEWEB = ROOT / "data/interim/pretrain/fineweb_extension_recovered.jsonl.gz"
FINEWEB_MANIFEST = ROOT / "data/manifests/pretrain/fineweb_extension_recovery_production.json"
WIKIMEDIA = ROOT / "data/processed/pretrain/factual/wikimedia_likelihood_source_v1/documents.jsonl"
WIKIMEDIA_MANIFEST = ROOT / "data/manifests/factual/wikimedia_likelihood_source_v1.json"
TOKENIZER = ROOT / "assets/tokenizer.json"


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest_timestamp(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    for field in ("acquisition_timestamp", "completed_at", "started_at"):
        observed = value.get(field)
        if isinstance(observed, str) and observed.strip():
            return observed
    raise ValueError(f"manifest has no acquisition timestamp: {path}")


def _source_specs(wikimedia_retrieved_at: str) -> list[dict[str, object]]:
    return [
        {
            "source_id": "fineweb_edu_extension_2025_26",
            "source_name": "FineWeb-Edu extension 2025-26",
            "license_name": "ODC-By 1.0",
            "license_url": "https://opendatacommons.org/licenses/by/1-0/",
            "documents": iter_fineweb_documents(FINEWEB),
        },
        {
            "source_id": "wikipedia_en_20231101",
            "source_name": "English Wikipedia 2023-11-01 snapshot",
            "license_name": "CC BY-SA 4.0 and GFDL",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "documents": iter_wikimedia_documents(WIKIMEDIA, wikimedia_retrieved_at),
        },
    ]


def build(recipient: str) -> dict[str, object]:
    if not recipient.startswith("age1"):
        raise ValueError("recipient must be an Age X25519 public recipient")
    for path in (FINEWEB, FINEWEB_MANIFEST, WIKIMEDIA, WIKIMEDIA_MANIFEST, TOKENIZER):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = ROOT / OUTPUT_ROOT
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    commit = _head()
    manifests: list[dict[str, object]] = []
    reservations: dict[str, dict[str, list[str]]] = {}
    for spec in _source_specs(_manifest_timestamp(WIKIMEDIA_MANIFEST)):
        source_id = str(spec["source_id"])
        development, held_out = reserve_documents(spec["documents"], source_id)
        reservations[source_id] = {
            "development": [item["parent_document_id"] for item in development],
            "held_out": [item["parent_document_id"] for item in held_out],
        }
        for split, documents in (("development", development), ("held_out", held_out)):
            payloads, provenance, contamination = build_records(
                tokenizer=tokenizer,
                source_id=source_id,
                source_name=str(spec["source_name"]),
                license_name=str(spec["license_name"]),
                license_url=str(spec["license_url"]),
                split=split,
                documents=documents,
            )
            manifests.append(
                build_inventory(
                    repository_root=ROOT,
                    relative_output=f"{OUTPUT_ROOT}/{source_id}/{split}",
                    suite_id=SUITE_ID,
                    repository_commit=commit,
                    source_id=source_id,
                    split=split,
                    recipient=recipient,
                    payloads=payloads,
                    provenance=provenance,
                    contamination=contamination,
                )
            )

    overlap = {
        source_id: sorted(set(value["development"]) & set(value["held_out"]))
        for source_id, value in reservations.items()
    }
    if any(overlap.values()):
        raise ValueError("development and held-out parent reservations overlap")
    report: dict[str, object] = {
        "schema_id": "vasu_140m_likelihood_self_curated_qualification_v1",
        "suite_id": SUITE_ID,
        "repository_commit": commit,
        "source_artifacts": {
            "fineweb": {
                "path": FINEWEB.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(FINEWEB),
                "manifest_sha256": sha256_file(FINEWEB_MANIFEST),
            },
            "wikimedia": {
                "path": WIKIMEDIA.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(WIKIMEDIA),
                "manifest_sha256": sha256_file(WIKIMEDIA_MANIFEST),
            },
        },
        "tokenizer_sha256": sha256_file(TOKENIZER),
        "counts": {
            source_id: {"development": ITEMS_PER_SPLIT, "held_out": ITEMS_PER_SPLIT}
            for source_id in reservations
        },
        "reservation_sha256s": {
            source_id: hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for source_id, value in reservations.items()
        },
        "inventory_sha256s": {
            f"{manifest['dimension']}:{manifest['split']}:{manifest['inventory_id']}": manifest["inventory_sha256"]
            for manifest in manifests
        },
        "parent_overlap_counts": {source_id: len(value) for source_id, value in overlap.items()},
        "recipient_fingerprint": hashlib.sha256(recipient.encode()).hexdigest(),
        "independently_curated": False,
        "fixture_only": True,
        "production_suite_frozen": False,
        "held_out_plaintext_persisted": False,
        "model_invoked": False,
        "checkpoint_accessed": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }
    report["qualification_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    qualification_path = output / "qualification.json"
    if qualification_path.exists():
        raise FileExistsError(f"qualification already exists: {qualification_path}")
    qualification_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipient", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.recipient), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
