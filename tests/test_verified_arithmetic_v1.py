from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_vasu_verified_arithmetic_v1 import build_corpus, verify_record
from vasu.data.mixtures.manifest import load_manifest
from vasu.data.mixtures.sampler import build_sampling_plan


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_arithmetic_corpus_is_deterministic_and_split_safe(tmp_path: Path) -> None:
    first = build_corpus(tmp_path / "first", {"train": 14, "development": 7, "evaluation": 7}, 42)
    second = build_corpus(tmp_path / "second", {"train": 14, "development": 7, "evaluation": 7}, 42)
    first_ids = set()
    for split in ("train", "development", "evaluation"):
        records = _records(Path(first["splits"][split]["path"]))
        assert first["splits"][split]["sha256"] == second["splits"][split]["sha256"]
        assert not first_ids.intersection(record["id"] for record in records)
        first_ids.update(record["id"] for record in records)
        assert all("7 + 8" not in str(record["text"]) for record in records)
        assert all("12 * 3" not in str(record["text"]) for record in records)
        lower, upper = {
            "train": (100, 999),
            "development": (1_000, 1_499),
            "evaluation": (1_500, 1_999),
        }[split]
        for record in records:
            operands = record["operand_metadata"]
            primary = operands.get("left", operands.get("start", operands.get("quotient")))
            if primary is not None:
                assert lower <= primary <= upper
        assert {record["template_id"] for record in records} == {
            "question_answer_v1"
        }


def test_arithmetic_answers_are_exactly_verifiable(tmp_path: Path) -> None:
    manifest = build_corpus(tmp_path, {"train": 28, "development": 7, "evaluation": 7}, 11)
    records = _records(Path(manifest["splits"]["train"]["path"]))
    for record in records:
        assert verify_record(record)
        assert record["text"] == (
            f"Question: {record['prompt']}\nAnswer: {record['answer']}"
        )
        assert "[EOS]" not in str(record["text"])
        assert record["generator_version"] == "vasu_verified_arithmetic_v1"
        assert record["template_id"] == "question_answer_v1"
        assert record["difficulty_tier"] == "tier_1"
        assert record["operand_metadata"]


def test_capability_mixture_plans_have_exact_budgets_and_safe_factual_shares() -> None:
    root = Path("configs/data/mixtures")
    for name in (
        "vasu_60m_capability_a_factual_20m.json",
        "vasu_60m_capability_b_balanced_20m.json",
        "vasu_60m_capability_c_control_20m.json",
    ):
        manifest = load_manifest(root / name)
        plan = build_sampling_plan(manifest)
        assert plan.allocated_tokens == 20_004_864
        factual = plan.allocation_for("wikimedia_factual_release_6aa10d73")
        assert factual.allocated_tokens <= factual.available_tokens
        assert not factual.replacement_required
