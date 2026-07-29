"""Exercise Candidate E paired-release validation using disposable fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from vasu.data.arithmetic_v2 import generate_records
from vasu.data.candidate_e_release import build_paired_release, validate_paired_release


def _fixture_splits() -> dict[str, list[dict]]:
    counts = {"train": 12, "development": 4, "evaluation": 4}
    return {
        split: [
            {**record, "id": f"candidate_e_fixture_v1:{split}:{index:07d}"}
            for index, record in enumerate(
                generate_records(split, 220, 20260729)[:count]
            )
        ]
        for split, count in counts.items()
    }


def run_fixture(tokenizer_path: Path) -> dict:
    """Build, validate, and automatically discard a tiny paired release."""

    with tempfile.TemporaryDirectory(prefix="vasu-candidate-e-fixture-") as temporary:
        root = Path(temporary)
        release = build_paired_release(
            logical_splits=_fixture_splits(),
            tokenizer_path=tokenizer_path,
            output_dir=root / "release",
            created_at="2026-07-29T00:00:00+00:00",
        )
        validated = validate_paired_release(
            root / "release", tokenizer_path=tokenizer_path
        )
        return {
            "fixture_only": True,
            "training_authorized": False,
            "arms": sorted(validated["arms"]),
            "source_split_counts": release["source_split_counts"],
        }


if __name__ == "__main__":
    print(
        json.dumps(run_fixture(Path("assets/tokenizer.json")), indent=2, sort_keys=True)
    )
