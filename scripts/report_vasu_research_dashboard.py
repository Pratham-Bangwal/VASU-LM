"""Print a read-only VASU research dashboard from repository evidence."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.report_candidate_e_governance_preflight import build_report
from vasu.utils.lineage import build_lineage_index


def build_dashboard(repository_root: Path) -> dict:
    lineage = build_lineage_index(repository_root)
    results = repository_root / "evaluation/results"
    result_directories = (
        sorted(item.name for item in results.iterdir() if item.is_dir())
        if results.is_dir()
        else []
    )
    by_category: dict[str, int] = {}
    for artifact in lineage["artifacts"]:
        by_category[artifact["category"]] = by_category.get(artifact["category"], 0) + 1
    return {
        "format_version": "vasu_research_dashboard_v1",
        "read_only": True,
        "lineage": {
            "artifact_count": lineage["artifact_count"],
            "by_category": by_category,
        },
        "evaluation_result_directories": result_directories,
        "candidate_e": build_report(repository_root),
    }


if __name__ == "__main__":
    print(json.dumps(build_dashboard(Path.cwd()), indent=2, sort_keys=True))
