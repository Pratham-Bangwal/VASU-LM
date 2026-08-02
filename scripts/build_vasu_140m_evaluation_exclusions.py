"""Freeze training exclusions from the self-curated likelihood qualification."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_base_v2 import canonical_json, sha256_file  # noqa: E402
from vasu.data.vasu_140m_evaluation_exclusions import (  # noqa: E402
    SCHEMA_ID,
    registry_identity,
    validate_exclusion_registry_files,
)


SUITE = ROOT / "evaluation/fixtures/vasu_140m_likelihood_self_curated_v1"
OUTPUT = ROOT / "configs/data/exclusions/vasu_140m_likelihood_self_curated_v1.json"
SOURCES = {
    "fineweb_edu_extension_2025_26": {
        "directory": "fineweb_edu_extension_2025_26",
        "admission": "configs/data/admissions/fineweb_edu_extension_2025_26.pending.json",
    },
    "wikipedia_en_20231101": {
        "directory": "wikipedia_en_20231101",
        "admission": "configs/data/admissions/wikipedia_en_20231101.pending.json",
    },
}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parents(manifest: dict[str, object]) -> list[str]:
    path = ROOT / manifest["provenance_index"]["path"]
    return [
        str(json.loads(line)["parent_document_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sources = []
    for source_id, spec in SOURCES.items():
        bindings = {}
        parents: set[str] = set()
        for split in ("development", "held_out"):
            path = SUITE / spec["directory"] / split / "manifest.json"
            manifest = _json(path)
            split_parents = set(_parents(manifest))
            if parents & split_parents:
                raise ValueError("likelihood split parents overlap")
            parents.update(split_parents)
            bindings[f"{split}_manifest"] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
        ids = sorted(parents)
        admission_path = ROOT / spec["admission"]
        sources.append(
            {
                "source_id": source_id,
                "pending_admission": {
                    "path": spec["admission"],
                    "sha256": sha256_file(admission_path),
                },
                **bindings,
                "excluded_parent_document_ids": ids,
                "excluded_parent_count": len(ids),
                "exclusion_sha256": hashlib.sha256(canonical_json(ids)).hexdigest(),
            }
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    registry: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "registry_id": "vasu-140m-likelihood-self-curated-v1-exclusions",
        "suite_id": "vasu-140m-base-evaluation-v2-likelihood-self-curated-v1",
        "repository_commit": commit,
        "sources": sources,
        "fixture_only": True,
        "independently_curated": False,
        "production_suite_frozen": False,
        "training_data_release_created": False,
        "training_authorized": False,
    }
    registry["registry_sha256"] = registry_identity(registry)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_exclusion_registry_files(registry, ROOT)
    return registry


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
