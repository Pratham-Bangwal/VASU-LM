"""Build the two review-ready source-admission v4 candidate packages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vasu.data.vasu_140m_admission_candidates import build_candidate, write_new  # noqa: E402

OUTPUT = ROOT / "configs/data/admissions/candidates"


def main() -> None:
    specs = {
        "fineweb_edu_extension_2025_26.v4.candidate.json": (
            Path("configs/data/admissions/fineweb_edu_extension_2025_26.pending.json"), True
        ),
        "wikipedia_en_20231101.v4.candidate.json": (
            Path("configs/data/admissions/wikipedia_en_20231101.pending.json"), False
        ),
    }
    results = {}
    for name, (pending, quarantine) in specs.items():
        package = build_candidate(ROOT, pending, require_fineweb_quarantine=quarantine)
        write_new(package, OUTPUT / name)
        results[name] = package["package_sha256"]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
