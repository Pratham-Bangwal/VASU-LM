"""Reproduce the clean-commit VASU-140M qualification identity."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_production_release import (  # noqa: E402
    qualify_production_release,
    validate_qualification_report,
)


IMPLEMENTATION_COMMIT = "c014716ec38ef8f08842356fc016359dc5a233d7"
FROZEN_REPORT = (
    REPOSITORY_ROOT
    / "evaluation/fixtures/"
    "vasu_140m_instruction_seed_v1_production_qualification_"
    "postcommit_c014716.json"
)


def main() -> None:
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if observed_commit != IMPLEMENTATION_COMMIT:
        raise ValueError("HEAD does not match the reviewed implementation commit")
    expected = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    observed = qualify_production_release(
        repository_root=REPOSITORY_ROOT,
        repository_commit=IMPLEMENTATION_COMMIT,
    ).qualification
    validate_qualification_report(expected)
    validate_qualification_report(observed)
    if observed != expected:
        raise ValueError("post-commit qualification does not match frozen evidence")
    print(json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()
