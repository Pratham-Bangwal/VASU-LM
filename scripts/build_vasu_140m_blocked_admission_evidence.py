"""Freeze locally complete but independently blocked source admission evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_base_v2 import sha256_file  # noqa: E402
from vasu.data.vasu_140m_blocked_admission import (  # noqa: E402
    SCHEMA_ID, evidence_identity, validate_blocked_admission_files,
)


SPECS = {
    "fineweb_edu_extension_2025_26": {
        "pending": "configs/data/admissions/fineweb_edu_extension_2025_26.pending.json",
        "artifact": "data/interim/pretrain/fineweb_extension_recovered.jsonl.gz",
        "manifest": "data/manifests/pretrain/fineweb_extension_recovery_production.json",
        "scan": "evaluation/results/vasu_140m_fineweb_exact_contamination_v2_20260802.json",
    },
    "wikipedia_en_20231101": {
        "pending": "configs/data/admissions/wikipedia_en_20231101.pending.json",
        "artifact": "data/processed/pretrain/factual/wikimedia_likelihood_source_v1/documents.jsonl",
        "manifest": "data/manifests/factual/wikimedia_likelihood_source_v1.json",
        "scan": "evaluation/results/vasu_140m_wikimedia_exact_contamination_v2_20260802.json",
    },
}
EXCLUSIONS = "configs/data/exclusions/vasu_140m_likelihood_self_curated_v1.json"


def binding(relative: str) -> dict[str, str]:
    return {"path": relative, "sha256": sha256_file(ROOT / relative)}


def main() -> None:
    output = ROOT / "configs/data/admissions/blocked_evidence"
    output.mkdir(parents=True, exist_ok=True)
    for source_id, spec in SPECS.items():
        value: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "evidence_id": f"vasu-140m-{source_id}-blocked-evidence-v1",
            "source_id": source_id,
            "pending_admission": binding(spec["pending"]),
            "source_artifact": binding(spec["artifact"]),
            "source_manifest": binding(spec["manifest"]),
            "evaluation_exclusions": binding(EXCLUSIONS),
            "exact_contamination_scan": binding(spec["scan"]),
            "remaining_blockers": [
                "independently curated production prompt inventory matrix",
                "independent semantic candidate search and review",
            ],
            "decision": "blocked",
            "source_admission_approved": False,
            "release_build_permitted": False,
            "training_authorized": False,
        }
        value["evidence_sha256"] = evidence_identity(value)
        path = output / f"{source_id}.blocked.json"
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        validate_blocked_admission_files(value, ROOT)
        print(source_id, value["evidence_sha256"])


if __name__ == "__main__":
    main()
