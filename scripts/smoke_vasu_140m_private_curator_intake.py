"""Qualify private-curator intake validation with synthetic temporary inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_private_curator_intake import (  # noqa: E402
    APPROVED_STATUS,
    COUNTS,
    FILENAMES,
    RECORD_SCHEMA_ID,
    validate_private_curator_intake,
)


def _sha(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _provenance() -> dict[str, str]:
    return {
        "source_name": "Synthetic independent-curator qualification",
        "source_url": "https://example.org/qualification",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "fixture-v1",
        "citation": "Synthetic temporary qualification input.",
        "authored_by": "qualification-fixture",
    }


def _record(dimension: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": RECORD_SCHEMA_ID,
        "status": APPROVED_STATUS,
        "item_id": f"qualification-{dimension}-001",
        "dimension": dimension,
        "semantic_family_id": f"qualification-family-{dimension}-001",
        "parent_document_id": f"qualification-parent-{dimension}-001",
        "provenance": _provenance(),
    }
    if dimension == "factuality":
        value.update(
            prompt="Qualification question about a synthetic mineral?",
            choices=[
                {"choice_id": "a", "text": "Option alpha"},
                {"choice_id": "b", "text": "Option beta"},
            ],
            correct_choice_id="a",
        )
    elif dimension == "arithmetic":
        value.update(
            prompt="Calculate exactly: 43 * 17 = ?",
            answer_type="integer",
            expected_answer="731",
        )
    elif dimension == "repetition":
        value.update(
            prompt="A synthetic qualification scene continued beyond the quiet archway",
            loop_ngram_size=3,
        )
    elif dimension == "robustness":
        value.update(
            baseline_prompt="Synthetic baseline asks for seven units.",
            variant_prompt="synthetic VARIANT asks for exactly seven units!",
            accepted_answers=["7", "seven"],
        )
    else:
        value.update(
            prompt="Explain a synthetic qualification process clearly.",
            rubric_dimensions=["coherence", "factual_support", "degeneration"],
        )
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def qualify() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="vasu_curator_intake_") as temporary:
        temporary_root = Path(temporary)
        repository = temporary_root / "repository"
        private = temporary_root / "private"
        private.mkdir()
        suite = (
            repository
            / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
        )
        for dimension in COUNTS:
            development = suite / dimension
            development.mkdir(parents=True)
            content = (
                {
                    "baseline_prompt": f"Development baseline {dimension}",
                    "variant_prompt": f"Development variant {dimension}",
                }
                if dimension == "robustness"
                else {"prompt": f"Development prompt {dimension}"}
            )
            _write(development / "payload.jsonl", {"content": content})
            _write(private / FILENAMES[dimension], _record(dimension))
        intake = validate_private_curator_intake(
            repository,
            private,
            expected_counts={dimension: 1 for dimension in COUNTS},
        )
    return {
        "schema_id": "vasu_140m_private_curator_intake_qualification_v1",
        "implementation_sha256": _sha(
            ROOT / "evaluation/framework/vasu_140m_private_curator_intake.py"
        ),
        "intake_report_sha256": intake["report_sha256"],
        "dimensions_validated": sorted(COUNTS),
        "records_validated": intake["total_records"],
        "development_overlap_count": intake["development_overlap_count"],
        "plaintext_copied_to_repository": intake[
            "plaintext_copied_to_repository"
        ],
        "private_key_opened": intake["private_key_opened"],
        "held_out_opening_authorized": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    print(json.dumps(qualify(), indent=2, sort_keys=True))
