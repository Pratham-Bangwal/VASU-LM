"""Create the immutable Batch 002 release after full human review."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "e62bac5680362164a0b898c8356d700363de024f05bfcd3bdf981c36f737b241"
)

SOURCE = Path(
    "data/raw/instruct/"
    "vasu_instruction_quality_v1_batch_002.jsonl"
)

REVIEW_MANIFEST = Path(
    "data/reviews/instruct/"
    "vasu_instruction_quality_v1_batch_002_final_full_review_manifest.json"
)

RELEASE_DIR = Path(
    "data/released/instruct/vasu_instruction_quality_v1_batch_002"
)

RELEASE_JSONL = RELEASE_DIR / (
    "vasu_instruction_quality_v1_batch_002.jsonl"
)

RELEASE_MANIFEST = RELEASE_DIR / "release_manifest.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source: {SOURCE}")

    if not REVIEW_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing full-review manifest: {REVIEW_MANIFEST}"
        )

    source_sha256 = sha256_file(SOURCE)

    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "Source changed after review. "
            f"Expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )

    review = json.loads(
        REVIEW_MANIFEST.read_text(encoding="utf-8")
    )

    required_review_values = {
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "source_record_count": 500,
        "approved_count": 500,
        "rejected_count": 0,
        "full_review_passed": True,
        "review_status": "passed",
    }

    for key, expected in required_review_values.items():
        actual = review.get(key)

        if actual != expected:
            raise ValueError(
                f"Review manifest mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}"
            )

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(SOURCE, RELEASE_JSONL)

    release_sha256 = sha256_file(RELEASE_JSONL)

    if release_sha256 != source_sha256:
        raise RuntimeError("Released JSONL does not match reviewed source")

    manifest = {
        "schema_version": (
            "vasu_instruction_quality_release_manifest_v1"
        ),
        "dataset_name": (
            "vasu_instruction_quality_v1_batch_002"
        ),
        "released_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_path": SOURCE.as_posix(),
        "source_sha256": source_sha256,
        "source_record_count": 500,
        "full_review_manifest_path": (
            REVIEW_MANIFEST.as_posix()
        ),
        "full_review_manifest_sha256": (
            sha256_file(REVIEW_MANIFEST)
        ),
        "release_path": RELEASE_JSONL.as_posix(),
        "release_sha256": release_sha256,
        "release_record_count": 500,
        "release_authorized": True,
        "tokenization_authorized": True,
        "training_authorized": False,
        "immutable": True,
        "next_required_gate": (
            "tokenization_and_response_mask_validation"
        ),
    }

    RELEASE_MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Release JSONL: {RELEASE_JSONL}")
    print(f"Release manifest: {RELEASE_MANIFEST}")
    print(f"Release SHA-256: {release_sha256}")
    print("Release records: 500")
    print("Release authorized: True")
    print("Tokenization authorized: True")
    print("Training authorized: False")


if __name__ == "__main__":
    main()
