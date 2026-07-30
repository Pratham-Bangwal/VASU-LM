"""Reproduce the frozen post-commit v2 authorization-gate qualification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.smoke_vasu_140m_authorization_protocol import main as emit_report  # noqa: E402


IMPLEMENTATION_COMMIT = "050d1fa39ee288d6be2c1edc2acca5cb1ddab81f"
FROZEN_REPORT = (
    REPOSITORY_ROOT
    / "evaluation/fixtures/"
    "vasu_140m_authorization_protocol_v2_qualification_postcommit_050d1fa.json"
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
        raise ValueError("HEAD does not match the reviewed v2 implementation commit")
    expected = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    # Keep the underlying smoke as the sole report constructor.
    from io import StringIO
    from contextlib import redirect_stdout

    output = StringIO()
    with redirect_stdout(output):
        emit_report()
    observed = json.loads(output.getvalue())
    if observed != expected:
        raise ValueError("post-commit v2 qualification does not match frozen evidence")
    print(json.dumps(observed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
