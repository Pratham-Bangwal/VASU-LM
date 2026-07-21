from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Callable

import pytest

from scripts.review_wikimedia_pilot import (
    ReviewError,
    ReviewValidationError,
    build_previews,
    generate_review,
    load_dataset,
    load_review_config,
    review_summary,
    select_chunks,
    set_review_status,
    validate_review,
    verify_input_hash,
)


FIXED_TIME = "2026-07-18T00:00:00+00:00"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(index: int, token_count: int) -> dict[str, object]:
    text = f"Chunk {index} factual text. " + (f"detail-{index} " * (index + 2))
    return {
        "format_version": "wikimedia_pilot_document_v2",
        "document_id": f"parent-{index // 3}:{index:04d}",
        "parent_document_id": f"parent-{index // 3}",
        "chunk_id": f"parent-{index // 3}:{index:04d}",
        "chunk_index": index % 3,
        "chunk_count": 3,
        "section_title": None,
        "source_url": f"https://example.test/article/{index // 3}",
        "title": f"Article {index // 3}",
        "token_count": token_count,
        "normalized_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "encoding_repaired": False,
        "quality_warnings": [],
        "filtering_metadata": {
            "encoding_repaired": False,
            "quality_warnings": [],
            "repeated_line_ratio": 0.0,
        },
        "provenance_metadata": {
            "source_row_index": index // 3,
            "parent_document_id": f"parent-{index // 3}",
        },
        "cleaned_text": text,
    }


def _environment(
    tmp_path: Path,
    *,
    mutate: Callable[[list[dict[str, object]]], None] | None = None,
) -> tuple[Path, list[dict[str, object]]]:
    records = [_record(index, 5 + index * 35) for index in range(30)]
    records[5]["quality_warnings"] = ["unusual_spacing"]
    records[6]["section_title"] = "References"
    records[7]["encoding_repaired"] = True
    if mutate is not None:
        mutate(records)
    input_path = tmp_path / "documents.jsonl"
    input_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    input_hash = _sha256(input_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "accepted_chunks": len(records),
                "accepted_parent_documents": len(
                    {str(record["parent_document_id"]) for record in records}
                ),
                "output_artifact_hashes": {"documents_jsonl_sha256": input_hash},
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "review.json"
    config_path.write_text(
        json.dumps(
            {
                "format_version": "wikimedia_manual_review_config_v1",
                "input_jsonl": str(input_path),
                "input_sha256": input_hash,
                "manifest_path": str(manifest_path),
                "output_json": str(tmp_path / "report.json"),
                "output_text": str(tmp_path / "report.txt"),
                "random_seed": 42,
                "random_sample_count": 5,
                "shortest_sample_count": 3,
                "longest_sample_count": 3,
                "near_max_sample_count": 3,
                "evenly_spaced_sample_count": 4,
                "distinct_parent_sample_count": 4,
                "preview_start_characters": 20,
                "preview_end_characters": 10,
                "minimum_token_threshold": 10,
                "maximum_token_count": 1024,
            }
        ),
        encoding="utf-8",
    )
    return config_path, records


def _selection(config_path: Path):
    config = load_review_config(config_path)
    chunks = load_dataset(config)
    return config, chunks, select_chunks(chunks, config)


def _mark_all_pass(config_path: Path) -> None:
    report = json.loads((config_path.parent / "report.json").read_text(encoding="utf-8"))
    for sample in report["samples"]:
        set_review_status(
            config_path,
            sample["chunk_id"],
            "pass",
            "Reviewed.",
            updated_at=FIXED_TIME,
        )


def test_input_sha256_verification(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    config = load_review_config(config_path)
    assert verify_input_hash(config) == config.input_sha256
    config.input_jsonl.write_text("tampered", encoding="utf-8")
    with pytest.raises(ReviewError, match="SHA-256 mismatch"):
        verify_input_hash(config)


def test_deterministic_random_selection(tmp_path: Path) -> None:
    config, chunks, first = _selection(_environment(tmp_path)[0])
    second = select_chunks(chunks, config)
    first_random = [c.chunk_id for c in first.selected if "random" in first.reasons[c.jsonl_line_number]]
    second_random = [c.chunk_id for c in second.selected if "random" in second.reasons[c.jsonl_line_number]]
    assert first_random == second_random


def test_sampling_does_not_mutate_global_random_state(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    state = random.getstate()
    _selection(config_path)
    assert random.getstate() == state


def test_shortest_selection(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    selected = [c.token_count for c in result.selected if "shortest" in result.reasons[c.jsonl_line_number]]
    assert selected == [5, 40, 75]


def test_longest_selection(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    selected = [c.token_count for c in result.selected if "longest" in result.reasons[c.jsonl_line_number]]
    assert selected == [950, 985, 1020]


def test_near_maximum_selection(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    selected = [c.token_count for c in result.selected if "near_maximum" in result.reasons[c.jsonl_line_number]]
    assert selected == [950, 985, 1020]


def test_evenly_spaced_selection(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    lines = [c.jsonl_line_number for c in result.selected if "evenly_spaced" in result.reasons[c.jsonl_line_number]]
    assert lines == [1, 11, 20, 30]


def test_distinct_parent_selection(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    parents = [c.parent_document_id for c in result.selected if "distinct_parent" in result.reasons[c.jsonl_line_number]]
    assert len(parents) == len(set(parents)) == 4


def test_quality_warning_chunks_are_forced_into_sample(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    assert any(c.jsonl_line_number == 6 and "quality_warning" in result.reasons[6] for c in result.selected)


def test_reference_sections_are_forced_into_sample(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    assert any(c.jsonl_line_number == 7 and "reference_section" in result.reasons[7] for c in result.selected)


def test_overlapping_selections_are_deduplicated(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    assert result.duplicate_selections_removed > 0
    assert len({chunk.chunk_id for chunk in result.selected}) == len(result.selected)


def test_sample_order_is_stable_jsonl_order(tmp_path: Path) -> None:
    _, _, result = _selection(_environment(tmp_path)[0])
    lines = [chunk.jsonl_line_number for chunk in result.selected]
    assert lines == sorted(lines)


def test_preview_lengths_are_bounded(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    assert all(len(sample["preview_start"]) <= 20 for sample in report["samples"])
    assert all(len(sample["preview_end"]) <= 10 for sample in report["samples"])


def test_report_never_contains_cleaned_text(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    assert all("cleaned_text" not in sample for sample in report["samples"])


def test_short_chunk_preview_is_not_duplicated() -> None:
    assert build_previews("short", 20, 10) == ("short", "")


def test_all_selection_reasons_are_recorded(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    overlapping = [sample for sample in report["samples"] if len(sample["selection_reasons"]) > 1]
    assert overlapping
    assert all(sample["selection_reasons"] for sample in report["samples"])


def test_status_update_is_atomic_and_preserves_sample(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    original_ids = [sample["chunk_id"] for sample in report["samples"]]
    updated = set_review_status(
        config_path,
        original_ids[0],
        "pass",
        "Clean.",
        updated_at=FIXED_TIME,
    )
    assert [sample["chunk_id"] for sample in updated["samples"]] == original_ids
    assert updated["samples"][0]["review_status"] == "pass"
    assert not (tmp_path / "report.json.tmp").exists()
    assert not (tmp_path / "report.txt.tmp").exists()


def test_invalid_status_is_rejected(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    with pytest.raises(ReviewError, match="Invalid review status"):
        set_review_status(config_path, report["samples"][0]["chunk_id"], "good", "")


def test_unknown_chunk_id_is_rejected(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    generate_review(config_path, generated_at=FIXED_TIME)
    with pytest.raises(ReviewError, match="Unknown sampled chunk ID"):
        set_review_status(config_path, "missing:0000", "pass", "")


def test_status_update_does_not_regenerate_selection(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    chunk_id = report["samples"][0]["chunk_id"]
    set_review_status(config_path, chunk_id, "pass", "Reviewed.", updated_at=FIXED_TIME)
    with pytest.raises(ReviewError, match="refusing to regenerate"):
        generate_review(config_path, generated_at=FIXED_TIME)
    persisted = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert persisted["samples"][0]["review_status"] == "pass"


def test_summary_status_counts_remain_correct(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    first = report["samples"][0]["chunk_id"]
    set_review_status(config_path, first, "minor_issue", "Small boundary issue.")
    summary = review_summary(config_path)
    assert summary["minor_issue_count"] == 1
    assert summary["pending_review_count"] == len(report["samples"]) - 1


def test_validation_fails_with_pending_records(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    generate_review(config_path, generated_at=FIXED_TIME)
    with pytest.raises(ReviewValidationError, match="review is pending"):
        validate_review(config_path)


def test_validation_fails_with_rejected_record(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    _mark_all_pass(config_path)
    set_review_status(config_path, report["samples"][0]["chunk_id"], "reject", "Bad text.")
    with pytest.raises(ReviewValidationError, match="review was rejected"):
        validate_review(config_path)


def test_validation_passes_with_pass_and_documented_minor_issue(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    _mark_all_pass(config_path)
    set_review_status(
        config_path,
        report["samples"][0]["chunk_id"],
        "minor_issue",
        "Non-systematic short chunk.",
    )
    assert validate_review(config_path)["summary"]["reject_count"] == 0


def test_missing_provenance_is_detected(tmp_path: Path) -> None:
    def mutate(records):
        records[0]["provenance_metadata"] = {}

    config_path, _ = _environment(tmp_path, mutate=mutate)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    sample = next(item for item in report["samples"] if item["jsonl_line_number"] == 1)
    assert "missing_parent_or_chunk_provenance" in sample["automatic_prechecks"]


def test_replacement_character_is_detected(tmp_path: Path) -> None:
    def mutate(records):
        records[0]["cleaned_text"] = "damaged \ufffd text"
        records[0]["normalized_sha256"] = hashlib.sha256("damaged \ufffd text".encode()).hexdigest()

    config_path, _ = _environment(tmp_path, mutate=mutate)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    sample = next(item for item in report["samples"] if item["jsonl_line_number"] == 1)
    assert "replacement_character" in sample["automatic_prechecks"]


def test_token_count_above_maximum_is_detected(tmp_path: Path) -> None:
    def mutate(records):
        records[0]["token_count"] = 1025

    config_path, _ = _environment(tmp_path, mutate=mutate)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    sample = next(item for item in report["samples"] if item["jsonl_line_number"] == 1)
    assert "token_count_above_maximum" in sample["automatic_prechecks"]


def test_duplicate_chunk_hash_is_detected(tmp_path: Path) -> None:
    def mutate(records):
        records[1]["normalized_sha256"] = records[0]["normalized_sha256"]

    config_path, _ = _environment(tmp_path, mutate=mutate)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    affected = [sample for sample in report["samples"] if sample["jsonl_line_number"] in {1, 2}]
    assert all("duplicate_chunk_hash" in sample["automatic_prechecks"] for sample in affected)


def test_no_current_working_directory_assumption(tmp_path: Path, monkeypatch) -> None:
    config_path, _ = _environment(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    assert report["summary"]["total_chunks_in_dataset"] == 30


def test_report_is_deterministic_except_timestamps(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    first = generate_review(config_path, generated_at="time-one")
    second = generate_review(config_path, force=True, generated_at="time-two")
    first_copy = copy.deepcopy(first)
    second_copy = copy.deepcopy(second)
    first_copy["summary"].pop("generation_timestamp")
    second_copy["summary"].pop("generation_timestamp")
    assert first_copy == second_copy


def test_text_report_contains_checklist_without_full_long_text(tmp_path: Path) -> None:
    config_path, records = _environment(tmp_path)
    generate_review(config_path, generated_at=FIXED_TIME)
    report_text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "[ ] No mojibake or replacement characters" in report_text
    longest_text = max((str(record["cleaned_text"]) for record in records), key=len)
    assert longest_text not in report_text


def test_generation_does_not_modify_input_dataset(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    config = load_review_config(config_path)
    before = _sha256(config.input_jsonl)
    generate_review(config_path, generated_at=FIXED_TIME)
    assert _sha256(config.input_jsonl) == before


def test_review_report_hash_mismatch_fails(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    generate_review(config_path, generated_at=FIXED_TIME)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["summary"]["input_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ReviewError, match="input hash does not match"):
        validate_review(config_path)


def test_regenerated_review_statuses_are_pending(tmp_path: Path) -> None:
    config_path, _ = _environment(tmp_path)
    report = generate_review(config_path, generated_at=FIXED_TIME)
    assert report["summary"]["pending_review_count"] == len(report["samples"])
    assert {sample["review_status"] for sample in report["samples"]} == {"pending"}
