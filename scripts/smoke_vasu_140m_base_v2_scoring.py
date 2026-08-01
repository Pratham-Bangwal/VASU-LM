"""Qualify VASU-140M evaluation-v2 scorers with prompt-free fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_base_v2 import sha256_file  # noqa: E402
from evaluation.framework.vasu_140m_base_v2_statistics import (  # noqa: E402
    build_fixture_qualification,
    qualification_evidence,
)
from evaluation.framework.vasu_140m_base_v2_tasks import (  # noqa: E402
    build_prompt_free_fixture_material,
)


SCHEMA_PATH = ROOT / "evaluation/framework/vasu_140m_base_v2.py"
TASK_PATH = ROOT / "evaluation/framework/vasu_140m_base_v2_tasks.py"
STATISTICS_PATH = ROOT / "evaluation/framework/vasu_140m_base_v2_statistics.py"
TEST_PATH = ROOT / "tests/test_vasu_140m_base_v2_scoring.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="emit the compact frozen identity instead of the complete report",
    )
    args = parser.parse_args()
    repository_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tasks, observations = build_prompt_free_fixture_material()
    report = build_fixture_qualification(
        tasks=tasks,
        observations=observations,
        repository_commit=repository_commit,
        schema_implementation_sha256=sha256_file(SCHEMA_PATH),
        task_implementation_sha256=sha256_file(TASK_PATH),
        statistics_implementation_sha256=sha256_file(STATISTICS_PATH),
        tests_sha256=sha256_file(TEST_PATH),
        smoke_sha256=sha256_file(Path(__file__)),
        bootstrap_samples=2_000,
        bootstrap_seed=140,
    )
    output = qualification_evidence(report) if args.evidence_only else report
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
