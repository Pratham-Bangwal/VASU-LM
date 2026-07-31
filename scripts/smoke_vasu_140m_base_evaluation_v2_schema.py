"""Emit deterministic, prompt-free qualification for evaluation-v2 schemas."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.framework.vasu_140m_base_v2 import (  # noqa: E402
    DIMENSIONS,
    build_synthetic_result_manifest,
    build_synthetic_suite_manifest,
    canonical_json,
    sha256_file,
    validate_result_manifest,
    validate_suite_manifest_files,
)


MODULE_PATH = Path("evaluation/framework/vasu_140m_base_v2.py")
TEST_PATH = Path("tests/test_vasu_140m_base_evaluation_v2.py")
SMOKE_PATH = Path("scripts/smoke_vasu_140m_base_evaluation_v2_schema.py")


def run() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    scorer_path = Path("evaluation/framework/scoring.py")
    development_path = TEST_PATH
    suite = build_synthetic_suite_manifest(
        repository_commit=commit,
        tokenizer_path="assets/tokenizer.json",
        tokenizer_sha256=sha256_file(REPOSITORY_ROOT / "assets/tokenizer.json"),
        scorer_path=scorer_path.as_posix(),
        scorer_sha256=sha256_file(REPOSITORY_ROOT / scorer_path),
        development_inventory_path=development_path.as_posix(),
        development_inventory_sha256=sha256_file(
            REPOSITORY_ROOT / development_path
        ),
    )
    validate_suite_manifest_files(suite, REPOSITORY_ROOT)
    result = build_synthetic_result_manifest(
        repository_commit=commit,
        suite_sha256=str(suite["suite_sha256"]),
    )
    validate_result_manifest(result)
    report: dict[str, object] = {
        "schema_id": "vasu_140m_base_evaluation_v2_schema_qualification_v1",
        "repository_commit": commit,
        "module_sha256": sha256_file(REPOSITORY_ROOT / MODULE_PATH),
        "test_sha256": sha256_file(REPOSITORY_ROOT / TEST_PATH),
        "smoke_sha256": sha256_file(REPOSITORY_ROOT / SMOKE_PATH),
        "synthetic_suite_sha256": suite["suite_sha256"],
        "synthetic_result_sha256": result["result_sha256"],
        "dimensions": sorted(DIMENSIONS),
        "development_files_verified": True,
        "held_out_files_opened": False,
        "real_prompts_frozen": False,
        "model_invoked": False,
        "checkpoint_opened": False,
        "result_published": False,
        "training_authorized": False,
    }
    report["qualification_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
