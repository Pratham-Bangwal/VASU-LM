from __future__ import annotations

import json
from pathlib import Path

from evaluation import prepare_ultrachat_parent_baseline as baseline


def test_baseline_uses_same_prompt_suite_and_expected_parent() -> None:
    payload = baseline.build_baseline(verify_checkpoint_hash=False)
    assert payload["checkpoint"].endswith(
        "alpaca_masked_v3_from_200k/best.pt"
    )
    assert payload["model_configuration"] == "vasu_60m"
    assert payload["prompt_format"] == "alpaca"
    assert payload["prompt_count"] == 40
    assert len(payload["prompt_ids"]) == len(set(payload["prompt_ids"]))
    assert set(payload["modes"]) == {"greedy", "sampled"}


def test_baseline_reports_required_metrics() -> None:
    payload = baseline.build_baseline(verify_checkpoint_hash=False)
    assert payload["masked_validation_loss"] == 2.522585
    for mode in payload["modes"].values():
        metrics = mode["metrics"]
        assert "general_instruction_following" in metrics
        assert "short_factual_questions" in metrics
        assert "format_following" in metrics
        overall = metrics["all_prompts"]
        assert "response_relevance" in overall
        assert "repetition" in overall
        assert "empty_output_rate" in overall
        assert "response_length_words" in overall
        assert "obvious_incoherence" in overall


def test_atomic_output_is_valid_and_reproducible(tmp_path: Path) -> None:
    payload = baseline.build_baseline(verify_checkpoint_hash=False)
    output = tmp_path / "baseline.json"
    baseline.atomic_write_json(output, payload)
    first = output.read_bytes()
    baseline.atomic_write_json(output, payload)
    assert output.read_bytes() == first
    assert json.loads(first)["prompt_count"] == 40
    assert not output.with_suffix(".json.tmp").exists()


def test_ultrachat_parent_does_not_reference_factual_cpt() -> None:
    source = Path(
        "train_vasu_60m_ultrachat_masked_v2_from_alpaca_v3.py"
    ).read_text(encoding="utf-8")
    assert "alpaca_masked_v3_from_200k/best.pt" in source
    assert "factual_cpt_wikimedia_15pct_from_200k" not in source


def test_factual_cpt_closeout_status_is_explicit() -> None:
    metadata = json.loads(
        Path(
            "checkpoints/vasu_60m/"
            "factual_cpt_wikimedia_15pct_from_200k/best.metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["completed_experiment"] is True
    assert metadata["promotion_status"] == "rejected"
    assert metadata["recommended_for_further_cpt"] is False
    assert metadata["recommended_for_instruction_tuning"] is False
    assert metadata["artifact_retention"] == "preserve"
