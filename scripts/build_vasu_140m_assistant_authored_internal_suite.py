"""Create the non-independent VASU-140M internal evaluation fixture."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from evaluation.framework.vasu_140m_assistant_authored_internal_suite import (
        build_suite,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    report = build_suite(
        repository_root=ROOT,
        output_directory=args.output_directory,
        repository_commit=commit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
