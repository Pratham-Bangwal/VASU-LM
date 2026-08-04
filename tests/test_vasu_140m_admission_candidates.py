from pathlib import Path

from vasu.data.vasu_140m_admission_candidates import build_candidate
from vasu.data.vasu_140m_source_admission_v4 import validate_admission_package_v4_files


def test_real_candidates_bind_ten_inventories_and_fineweb_quarantine() -> None:
    root = Path.cwd()
    fineweb = build_candidate(
        root,
        Path("configs/data/admissions/fineweb_edu_extension_2025_26.pending.json"),
        require_fineweb_quarantine=True,
    )
    wikipedia = build_candidate(
        root,
        Path("configs/data/admissions/wikipedia_en_20231101.pending.json"),
        require_fineweb_quarantine=False,
    )
    for package in (fineweb, wikipedia):
        validate_admission_package_v4_files(package, root)
        assert len(package["evaluation_isolation"]["inventories"]) == 10
        assert package["decision"]["state"] == "pending"
        assert package["training_authorized"] is False
    assert len(fineweb["mandatory_exclusions"]) == 1
    assert fineweb["mandatory_exclusions"][0]["required"] is True
    assert wikipedia["mandatory_exclusions"] == []
