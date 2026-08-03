from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_private_curator_intake import (
    APPROVED_STATUS,
    COUNTS,
    FILENAMES,
    RECORD_SCHEMA_ID,
    report_identity,
    validate_private_curator_intake,
)


SMALL_COUNTS = {dimension: 1 for dimension in COUNTS}
ROOT = Path(__file__).resolve().parents[1]


def _provenance() -> dict[str, str]:
    return {
        "source_name": "Independent curator authored material",
        "source_url": "https://example.org/curator-source",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "curator-v1",
        "citation": "Independently authored for sealed evaluation.",
        "authored_by": "independent-curator-fixture",
    }


def _record(dimension: str, ordinal: int = 1) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_id": RECORD_SCHEMA_ID,
        "status": APPROVED_STATUS,
        "item_id": f"heldout-{dimension}-{ordinal}",
        "dimension": dimension,
        "semantic_family_id": f"heldout-family-{dimension}-{ordinal}",
        "parent_document_id": f"heldout-parent-{dimension}-{ordinal}",
        "provenance": _provenance(),
    }
    if dimension == "factuality":
        base.update(
            prompt="Which metal is liquid near ordinary room temperature?",
            choices=[
                {"choice_id": "a", "text": "Mercury"},
                {"choice_id": "b", "text": "Iron"},
            ],
            correct_choice_id="a",
        )
    elif dimension == "arithmetic":
        base.update(
            prompt="Calculate exactly: 37 * 19 = ?",
            answer_type="integer",
            expected_answer="703",
        )
    elif dimension == "repetition":
        base.update(
            prompt="Across the quiet inlet the survey boat followed a marked route",
            loop_ngram_size=3,
        )
    elif dimension == "robustness":
        base.update(
            baseline_prompt="A hexagon has how many sides?",
            variant_prompt="how MANY sides does a hexagon have?",
            accepted_answers=["6", "six"],
        )
    else:
        base.update(
            prompt="Explain how a compass helps a traveler navigate.",
            rubric_dimensions=["coherence", "factual_support", "degeneration"],
        )
    return base


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
        newline="\n",
    )


def _workspace(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    repository = tmp_path / "repo"
    private = tmp_path / "private"
    private.mkdir()
    suite = repository / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
    records = {dimension: _record(dimension) for dimension in COUNTS}
    for dimension, record in records.items():
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
        _write_jsonl(development / "payload.jsonl", [{"content": content}])
        _write_jsonl(private / FILENAMES[dimension], [record])
    return repository, private, records


def test_valid_private_intake_emits_hashes_and_counts_only(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)
    report = validate_private_curator_intake(
        repository, private, expected_counts=SMALL_COUNTS
    )
    assert report["total_records"] == 5
    assert report["unique_item_ids"] == 5
    assert report["unique_semantic_families"] == 5
    assert report["unique_parent_documents"] == 5
    assert report["development_overlap_count"] == 0
    assert report["plaintext_copied_to_repository"] is False
    assert report["private_key_opened"] is False
    assert report["training_authorized"] is False
    assert report["report_sha256"] == report_identity(report)
    serialized = json.dumps(report)
    assert "Mercury" not in serialized
    assert "hexagon" not in serialized


def test_frozen_qualification_binds_current_implementation() -> None:
    report = json.loads(
        (
            ROOT
            / "evaluation/fixtures/"
            "vasu_140m_private_curator_intake_qualification_v1.json"
        ).read_text(encoding="utf-8")
    )
    implementation = (
        ROOT / "evaluation/framework/vasu_140m_private_curator_intake.py"
    ).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert report["implementation_sha256"] == hashlib.sha256(
        implementation.encode("utf-8")
    ).hexdigest()
    assert report["records_validated"] == 5
    assert report["development_overlap_count"] == 0
    assert report["plaintext_copied_to_repository"] is False
    assert report["private_key_opened"] is False
    assert report["training_authorized"] is False


def test_rejects_private_directory_inside_repository(tmp_path: Path) -> None:
    repository, _, _ = _workspace(tmp_path)
    inside = repository / "private"
    inside.mkdir()
    with pytest.raises(ValueError, match="outside repository"):
        validate_private_curator_intake(
            repository, inside, expected_counts=SMALL_COUNTS
        )


def test_rejects_schema_sample_status(tmp_path: Path) -> None:
    repository, private, records = _workspace(tmp_path)
    record = copy.deepcopy(records["factuality"])
    record["status"] = "schema_sample_only_not_for_evaluation"
    _write_jsonl(private / FILENAMES["factuality"], [record])
    with pytest.raises(ValueError, match="not approved for sealing"):
        validate_private_curator_intake(
            repository, private, expected_counts=SMALL_COUNTS
        )


def test_rejects_incorrect_arithmetic_answer(tmp_path: Path) -> None:
    repository, private, records = _workspace(tmp_path)
    record = copy.deepcopy(records["arithmetic"])
    record["expected_answer"] = "704"
    _write_jsonl(private / FILENAMES["arithmetic"], [record])
    with pytest.raises(ValueError, match="expected_answer is incorrect"):
        validate_private_curator_intake(
            repository, private, expected_counts=SMALL_COUNTS
        )


def test_rejects_development_prompt_overlap(tmp_path: Path) -> None:
    repository, private, records = _workspace(tmp_path)
    development = (
        repository
        / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
        / "factuality/payload.jsonl"
    )
    _write_jsonl(
        development,
        [{"content": {"prompt": records["factuality"]["prompt"]}}],
    )
    with pytest.raises(ValueError, match="overlaps development"):
        validate_private_curator_intake(
            repository, private, expected_counts=SMALL_COUNTS
        )


@pytest.mark.parametrize(
    "identity_field",
    ["item_id", "semantic_family_id", "parent_document_id"],
)
def test_rejects_cross_dimension_identity_reuse(
    tmp_path: Path, identity_field: str
) -> None:
    repository, private, records = _workspace(tmp_path)
    record = copy.deepcopy(records["repetition"])
    record[identity_field] = records["factuality"][identity_field]
    _write_jsonl(private / FILENAMES["repetition"], [record])
    with pytest.raises(ValueError, match=f"duplicate {identity_field}"):
        validate_private_curator_intake(
            repository, private, expected_counts=SMALL_COUNTS
        )


def test_rejects_wrong_record_count(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)
    (private / FILENAMES["manual_review"]).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="record count mismatch"):
        validate_private_curator_intake(
            repository, private, expected_counts=SMALL_COUNTS
        )


def test_rejects_linked_private_input_when_supported(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)
    target = private / FILENAMES["arithmetic"]
    replacement = private / "real-arithmetic.jsonl"
    target.replace(replacement)
    try:
        target.symlink_to(replacement)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    with pytest.raises(ValueError, match="missing or linked"):
        validate_private_curator_intake(
            repository, private, expected_counts=SMALL_COUNTS
        )
