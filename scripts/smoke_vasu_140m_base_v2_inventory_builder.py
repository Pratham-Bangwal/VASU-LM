"""Deterministic fixture-only qualification for the evaluation inventory builder."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.framework.vasu_140m_base_v2 import canonical_json  # noqa: E402
from evaluation.framework.vasu_140m_base_v2_inventory_builder import (  # noqa: E402
    AUTHORING_SCHEMA_ID,
    SUPPORTED_DIMENSIONS,
    build_fixture_inventory,
    build_fixture_report,
)
from evaluation.framework.vasu_140m_base_v2_inventory_plan import (  # noqa: E402
    validate_inventory_construction_plan_files,
)


PLAN_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/vasu_140m_base_v2_inventory_construction_plan.json"
)
SCORER_BYTES = b"# frozen fixture scorer\n"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance() -> dict[str, object]:
    return {
        "source_name": "VASU fixture authoring source",
        "source_url": "https://example.com/vasu-fixture-source",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "fixture-revision-1",
        "citation": "Project-authored synthetic inventory-builder fixture.",
        "retrieved_at": "2026-08-01T12:00:00+05:30",
        "human_authored": True,
    }


def _item(dimension: str) -> dict[str, object]:
    item_id = f"{dimension}-smoke-001"
    family = f"{dimension}-smoke-family-001"
    value: dict[str, object] = {
        "schema_id": AUTHORING_SCHEMA_ID,
        "item_id": item_id,
        "dimension": dimension,
        "semantic_family_id": family,
        "parent_document_id": f"{dimension}-smoke-parent-001",
        "provenance": _provenance(),
        "human_approved": True,
    }
    if dimension == "factuality":
        value.update(
            strata={"semantic_family": family, "task_family": "science"},
            content={
                "prompt": "Which planet is commonly called the Red Planet?",
                "choices": [
                    {"choice_id": "mars", "text": "Mars"},
                    {"choice_id": "venus", "text": "Venus"},
                ],
            },
            scoring={"correct_choice_id": "mars"},
        )
    elif dimension == "arithmetic":
        value.update(
            strata={
                "semantic_family": family,
                "operation": "addition",
                "difficulty": "fixture",
                "template_family": "fixture-exact-v1",
            },
            content={"prompt": "Calculate exactly: 1203 + 998 = ?"},
            scoring={"answer_type": "integer", "expected_answer": "2201"},
        )
    elif dimension == "repetition":
        value.update(
            strata={"semantic_family": family, "prompt_family": "narrative"},
            content={"prompt": "At the edge of the quiet forest, the old path continued"},
            scoring={"loop_ngram_size": 3},
        )
    elif dimension == "robustness":
        value.update(
            strata={
                "semantic_family": family,
                "variant_kind": "whitespace_and_case",
            },
            content={
                "baseline_prompt": "The capital of France is",
                "variant_prompt": "the   CAPITAL of France is",
            },
            scoring={"accepted_answers": ["Paris"]},
        )
    else:
        value.update(
            strata={"semantic_family": family, "category": "coherence"},
            content={"prompt": "A careful explanation of rainfall begins with"},
            scoring={
                "rubric_dimensions": [
                    "coherence",
                    "factual_support",
                    "degeneration",
                ]
            },
        )
    return value


def build_report() -> dict[str, object]:
    construction_plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    validate_inventory_construction_plan_files(construction_plan, REPOSITORY_ROOT)
    parent_commit = str(construction_plan["repository_commit"])
    with tempfile.TemporaryDirectory(
        prefix="vasu_140m_inventory_builder_smoke_"
    ) as raw_root:
        root = Path(raw_root)
        scorer_path = root / "evaluation/framework/scorer.py"
        scorer_path.parent.mkdir(parents=True)
        scorer_path.write_bytes(SCORER_BYTES)
        output_parent = root / "evaluation/fixtures/inventory_builder"
        output_parent.mkdir(parents=True)
        scorer = {
            "path": "evaluation/framework/scorer.py",
            "sha256": hashlib.sha256(SCORER_BYTES).hexdigest(),
        }
        manifests = [
            build_fixture_inventory(
                repository_root=root,
                output_directory=f"evaluation/fixtures/inventory_builder/{dimension}",
                inventory_id=f"fixture-{dimension}-v1",
                suite_id=str(construction_plan["suite_id"]),
                repository_commit=parent_commit,
                dimension=dimension,
                scorer=scorer,
                authoring_items=[_item(dimension)],
            )
            for dimension in sorted(SUPPORTED_DIMENSIONS)
        ]
        report = build_fixture_report(
            manifests,
            construction_plan=construction_plan,
        )
    report.update(
        {
            "repository_parent": parent_commit,
            "implementation_sha256": _sha256_file(
                REPOSITORY_ROOT
                / "evaluation/framework/vasu_140m_base_v2_inventory_builder.py"
            ),
            "test_sha256": _sha256_file(
                REPOSITORY_ROOT / "tests/test_vasu_140m_base_v2_inventory_builder.py"
            ),
            "temporary_artifacts_removed": not root.exists(),
        }
    )
    body = dict(report)
    body.pop("report_sha256", None)
    report["report_sha256"] = hashlib.sha256(canonical_json(body)).hexdigest()
    return report


def main() -> int:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["temporary_artifacts_removed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
