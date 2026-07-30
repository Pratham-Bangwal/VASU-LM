"""Run the read-only VASU-140M instruction seed release-plan qualification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_release_plan import (  # noqa: E402
    load_release_plan,
    validate_frozen_plan_report,
    validate_release_plan,
)


DEFAULT_PLAN = (
    REPOSITORY_ROOT
    / "configs"
    / "data"
    / "releases"
    / "vasu_140m_instruction_seed_v1.plan.json"
)
FROZEN_REPORT = (
    REPOSITORY_ROOT
    / "evaluation"
    / "fixtures"
    / "vasu_140m_instruction_seed_v1_plan_report.json"
)


def run(
    plan_path: Path = DEFAULT_PLAN,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    """Validate twice, compare frozen evidence, and create no output."""

    plan = load_release_plan(plan_path)
    first = validate_release_plan(plan, repository_root)
    second = validate_release_plan(plan, repository_root)
    if first != second:
        raise RuntimeError("release-plan qualification is not deterministic")
    frozen = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    if first != frozen:
        raise ValueError("release-plan qualification does not match frozen evidence")
    validate_frozen_plan_report(first)
    return first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.repository_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
