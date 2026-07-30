"""Reproduce the frozen read-only VASU-140M production qualification."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_production_release import (  # noqa: E402
    qualify_production_release,
    validate_qualification_report,
)


REVIEW_BASE_COMMIT = "20d79c3f1be58596c22b53c9ce87df7943b8a90c"
FROZEN_REPORT = (
    REPOSITORY_ROOT
    / "evaluation/fixtures/"
    "vasu_140m_instruction_seed_v1_production_qualification.json"
)


def main() -> None:
    expected = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    observed = qualify_production_release(
        repository_root=REPOSITORY_ROOT,
        repository_commit=REVIEW_BASE_COMMIT,
    ).qualification
    validate_qualification_report(expected)
    validate_qualification_report(observed)
    if observed != expected:
        raise ValueError("production qualification does not match frozen evidence")
    print(json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()
