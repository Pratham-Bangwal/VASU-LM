from __future__ import annotations

import json
from pathlib import Path

from vasu.data.vasu_140m_source_admission import validate_admission_package_files


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ROOT / "configs/data/admissions"


def test_pending_source_packages_are_bound_and_non_authorizing() -> None:
    paths = sorted(PACKAGES.glob("*.pending.json"))
    assert [path.name for path in paths] == [
        "fineweb_edu_extension_2025_26.pending.json",
        "wikipedia_en_20231101.pending.json",
    ]
    for path in paths:
        package = json.loads(path.read_text(encoding="utf-8"))
        validate_admission_package_files(package, ROOT)
        assert package["decision"]["state"] == "pending"
        assert package["acquisition"]["authorized"] is False
        assert package["legal_evidence"]["unresolved_items"]
        assert package["training_authorized"] is False
