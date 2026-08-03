"""Read-only pre-execution qualification for private curator sealing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_private_curator_intake import (  # noqa: E402
    validate_private_curator_intake,
)


RECEIPT = ROOT / "configs/evaluation/vasu_140m_private_curator_intake_20260803.json"


def _lf_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def qualify() -> dict[str, object]:
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    observed = validate_private_curator_intake(
        ROOT, Path(r"D:\VASU_PRIVATE_HELDOUT")
    )
    if observed["report_sha256"] != receipt["report_sha256"]:
        raise ValueError("private curator intake identity mismatch")
    age = subprocess.run(
        ["age", "--version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if age.returncode != 0:
        raise ValueError("Age executable is unavailable")
    recipient = str(receipt["recipient"])
    fingerprint = hashlib.sha256(recipient.encode("utf-8")).hexdigest()
    if fingerprint != receipt["recipient_fingerprint_sha256"]:
        raise ValueError("recipient fingerprint mismatch")
    return {
        "schema_id": "vasu_140m_private_curator_sealing_qualification_v1",
        "implementation_sha256": _lf_sha(
            ROOT / "evaluation/framework/vasu_140m_private_curator_sealing.py"
        ),
        "tests_sha256": _lf_sha(
            ROOT / "tests/test_vasu_140m_private_curator_sealing.py"
        ),
        "intake_receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
        "intake_report_sha256": observed["report_sha256"],
        "recipient_fingerprint_sha256": fingerprint,
        "age_available": True,
        "record_count": observed["total_records"],
        "sealing_invoked": False,
        "ciphertext_created": False,
        "private_key_opened": False,
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(qualify(), indent=2, sort_keys=True))
