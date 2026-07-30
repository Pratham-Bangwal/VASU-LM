"""Reproduce the frozen post-commit v2 authorization-gate qualification."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.smoke_vasu_140m_authorization_protocol import main as emit_report  # noqa: E402
from vasu.data.vasu_140m_records import sha256_json  # noqa: E402


IMPLEMENTATION_COMMIT = "050d1fa39ee288d6be2c1edc2acca5cb1ddab81f"
GATE_SHA256 = "d135889b89b424e7a3253adddec0b1cfd09b49efdb6831f048934f2468d9acac"
TEST_SHA256 = "ccda470cff74a2911cd7cbb4d5d993b0f6f7797b8da32c519cff341beebcc101"
SMOKE_SHA256 = "cbb2cd2ebb9692c4615ea16020d11cd5b0b7d2ad86c1d5facc1120af7a90506a"
FROZEN_REPORT = (
    REPOSITORY_ROOT
    / "evaluation/fixtures/"
    "vasu_140m_authorization_protocol_v2_qualification_postcommit_050d1fa.json"
)


def main() -> None:
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", IMPLEMENTATION_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )
    if ancestry.returncode != 0:
        raise ValueError("HEAD does not descend from the reviewed v2 implementation")
    identities = {
        "gate": (
            REPOSITORY_ROOT / "vasu/data/vasu_140m_authorization_protocol.py",
            GATE_SHA256,
        ),
        "tests": (
            REPOSITORY_ROOT / "tests/test_vasu_140m_authorization_protocol.py",
            TEST_SHA256,
        ),
        "smoke": (
            REPOSITORY_ROOT / "scripts/smoke_vasu_140m_authorization_protocol.py",
            SMOKE_SHA256,
        ),
    }
    for name, (path, expected_hash) in identities.items():
        observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError(f"reviewed {name} identity changed")
    expected = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    output = StringIO()
    with redirect_stdout(output):
        emit_report()
    observed = json.loads(output.getvalue())
    observed["repository_commit"] = IMPLEMENTATION_COMMIT
    observed.pop("qualification_sha256")
    observed["qualification_sha256"] = sha256_json(observed)
    if observed != expected:
        raise ValueError("post-commit v2 qualification does not match frozen evidence")
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
