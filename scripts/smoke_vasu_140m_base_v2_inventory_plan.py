"""Validate the frozen VASU-140M evaluation inventory construction plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.framework.vasu_140m_base_v2_inventory_plan import (  # noqa: E402
    validate_inventory_construction_plan_files,
)

PLAN_PATH = (
    REPOSITORY_ROOT
    / "configs/evaluation/vasu_140m_base_v2_inventory_construction_plan.json"
)


def build_report() -> dict[str, object]:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    validate_inventory_construction_plan_files(plan, REPOSITORY_ROOT)
    counts = {
        dimension: dict(item["counts"])
        for dimension, item in sorted(plan["dimensions"].items())
    }
    return {
        "schema_id": "vasu_140m_base_v2_inventory_plan_qualification_v1",
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "dependency_sha256s": {
            name: value["sha256"]
            for name, value in sorted(plan["dependencies"].items())
        },
        "accepted_dependencies_bound": 4,
        "dimension_counts": counts,
        "likelihood_count_scope": plan["dimensions"]["likelihood"]["count_scope"],
        "prompt_inventory_count_required_before_source_admission": 10,
        "likelihood_holdout_created_after_acquisition": True,
        "held_out_key_created": False,
        "held_out_content_created": False,
        "construction_authorized": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }


def main() -> int:
    print(json.dumps(build_report(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
