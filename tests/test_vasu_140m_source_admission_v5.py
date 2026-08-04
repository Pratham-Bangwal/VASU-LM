import hashlib
import json
from pathlib import Path

import pytest

from vasu.data.vasu_140m_source_admission import package_identity
from vasu.data.vasu_140m_source_admission_v5 import validate_admission_package_v5_files


def _package() -> dict[str, object]:
    root = Path.cwd()
    path = Path("configs/data/admissions/candidates/fineweb_edu_extension_2025_26.v4.candidate.json")
    package = json.loads((root / path).read_text(encoding="utf-8"))
    review = Path("docs/VASU_140M_SOURCE_ADMISSION_V4_CANDIDATES_INDEPENDENT_REVIEW_DECISION_20260804.md")
    package["schema_id"] = "vasu_140m_base_source_admission_v5"
    package["decision"] = {
        "state": "approved",
        "reviewed_by": "GPT-5.5 independent scientific reviewer",
        "reviewed_at": "2026-08-04T00:00:00+05:30",
        "notes": "Approved only for source admission; training remains unauthorized.",
    }
    package["source_admission_review"] = {
        "path": review.as_posix(),
        "sha256": hashlib.sha256((root / review).read_bytes()).hexdigest(),
        "decision": "accepted",
        "source_id": package["source_record"]["source_id"],
    }
    package["package_sha256"] = package_identity(package)
    return package


def test_v5_binds_real_source_review() -> None:
    validate_admission_package_v5_files(_package(), Path.cwd())


def test_v5_rejects_review_substitution() -> None:
    package = _package()
    package["source_admission_review"]["source_id"] = "another-source"
    package["package_sha256"] = package_identity(package)
    with pytest.raises(ValueError, match="source mismatch"):
        validate_admission_package_v5_files(package, Path.cwd())
