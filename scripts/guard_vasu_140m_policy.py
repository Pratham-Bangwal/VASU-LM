"""Fail CI when VASU-140M advances beyond its frozen readiness phase."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.training.vasu_140m_policy_guard import (  # noqa: E402
    build_policy_report,
    validate_policy_report,
)


def main() -> int:
    report = build_policy_report(REPOSITORY_ROOT)
    validate_policy_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
