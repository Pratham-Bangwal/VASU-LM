from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluation.evaluate_verified_arithmetic_v2 as evaluator_cli
from evaluation.verified_arithmetic import (
    GeneratedAnswer,
    classify_response,
    evaluate_records,
    format_arithmetic_prompt,
    load_verified_split,
    select_proxy_records,
    summarize_results,
)


def _record(
    identifier: str = "example",
    *,
    answer: str = "15",
    answer_type: str = "integer",
) -> dict[str, object]:
    return {
        "id": identifier,
        "split": "development",
        "operation": "addition",
        "difficulty_tier": "tier_1",
        "template_family": "heldout_exact",
        "answer_type": answer_type,
        "prompt": "Find 7 + 8.",
        "answer": answer,
    }


@pytest.mark.parametrize(
    ("response", "outcome"),
    [
        ("15", "correct"),
        ("14", "incorrect"),
        ("The answer is 15.", "malformed"),
        ("", "unanswered"),
        ("Question: Find 7 + 8.\nAnswer: 15", "prompt_leakage"),
    ],
)
def test_strict_outcome_classification(response: str, outcome: str) -> None:
    assert classify_response(_record(), response)["outcome"] == outcome


@pytest.mark.parametrize(
    ("answer_type", "answer"),
    [
        ("integer", "-12"),
        ("reduced_fraction", "-3/7"),
        ("comparison_symbol", ">"),
        ("boolean", "yes"),
    ],
)
def test_all_answer_types_are_exact(answer_type: str, answer: str) -> None:
    result = classify_response(
        _record(answer=answer, answer_type=answer_type),
        answer.upper(),
    )
    assert result["correct"]


def test_prompt_matches_training_boundary_without_answer() -> None:
    prompt = format_arithmetic_prompt(_record())
    assert prompt == "Question: Find 7 + 8.\nAnswer:"
    assert "15" not in prompt


def test_proxy_selection_is_deterministic_and_spread() -> None:
    records = [_record(str(index)) for index in range(10)]
    first = select_proxy_records(records, 4)
    second = select_proxy_records(records, 4)
    assert [item["id"] for item in first] == ["0", "2", "5", "7"]
    assert first == second


def test_evaluation_resume_skips_completed_ids_and_summarizes() -> None:
    records = [_record("a"), _record("b")]
    calls: list[str] = []

    def generate(record):
        calls.append(str(record["id"]))
        return GeneratedAnswer("15", 1, False, 0.25)

    results = evaluate_records(records, generate, completed_ids={"a"})
    assert calls == ["b"]
    summary = summarize_results(results)
    assert summary["overall"]["exact_accuracy"] == 1.0
    assert summary["by"]["operation"]["addition"]["count"] == 1
    assert summary["by"]["difficulty_tier"]["tier_1"]["correct"] == 1


def test_cli_persists_each_example_and_resumes_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    checkpoint_path = tmp_path / "checkpoint.pt"
    split_path = tmp_path / "dev.jsonl"
    for path in (manifest_path, tokenizer_path, checkpoint_path, split_path):
        path.write_text("fixture", encoding="utf-8")
    records = [_record("a"), _record("b")]
    manifest = {"tokenizer": {"sha256": "fixture-sha"}}
    monkeypatch.setattr(
        evaluator_cli,
        "load_verified_split",
        lambda manifest, split: (manifest_payload, records, split_path),
    )
    manifest_payload = manifest
    monkeypatch.setattr(evaluator_cli, "sha256_file", lambda path: "fixture-sha")
    monkeypatch.setattr(evaluator_cli.torch, "load", lambda *args, **kwargs: {"model": {}})

    class DummyModel:
        def load_state_dict(self, state, strict=True):
            return None

        def to(self, device):
            return self

        def eval(self):
            return self

    class DummyTokenizer:
        def load(self, path):
            return None

    monkeypatch.setattr(evaluator_cli, "VASUModel", lambda config: DummyModel())
    monkeypatch.setattr(evaluator_cli, "VASUTokenizer", DummyTokenizer)
    calls = 0

    def interrupted(records, generate, completed_ids=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupted")
        return [
            {
                "id": records[0]["id"],
                "operation": "addition",
                "difficulty_tier": "tier_1",
                "template_family": "heldout_exact",
                "answer_type": "integer",
                "outcome": "correct",
                "correct": True,
                "malformed": False,
                "unanswered": False,
                "prompt_leakage": False,
                "truncated": False,
                "repetition_ratio": 0.0,
                "duration_seconds": 0.1,
                "generated_tokens": 1,
            }
        ]

    monkeypatch.setattr(evaluator_cli, "evaluate_records", interrupted)
    args = SimpleNamespace(
        manifest=manifest_path,
        split="dev",
        tokenizer=tokenizer_path,
        checkpoint=checkpoint_path,
        max_new_tokens=16,
        output_dir=tmp_path / "output",
        resume=False,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluator_cli.run(args)
    saved = evaluator_cli._load_existing(args.output_dir / "per_example.jsonl")
    assert [item["id"] for item in saved] == ["a"]

    calls = 0
    args.resume = True
    evaluator_cli.run(args)
    saved = evaluator_cli._load_existing(args.output_dir / "per_example.jsonl")
    assert [item["id"] for item in saved] == ["a", "b"]


def test_manifest_hash_and_split_validation(tmp_path: Path) -> None:
    split = tmp_path / "dev.jsonl"
    row = _record()
    split.write_text(json.dumps(row) + "\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(split.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": "verified_arithmetic_v2",
        "artifacts": {
            "dev.jsonl": {"path": "dev.jsonl", "sha256": digest},
        },
        "logical_example_counts": {"development": 1},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, rows, loaded_path = load_verified_split(path, "dev")
    assert loaded_path == split
    assert rows[0]["id"] == "example"
    split.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_verified_split(path, "dev")


def test_production_v2_dev_and_eval_are_complete_and_distinct() -> None:
    manifest = Path(
        "data/processed/capability/verified_arithmetic_v2/manifest.json"
    )
    _, development, _ = load_verified_split(manifest, "dev")
    _, evaluation, _ = load_verified_split(manifest, "eval")
    assert len(development) == len(evaluation) == 1000
    assert {item["id"] for item in development}.isdisjoint(
        item["id"] for item in evaluation
    )
    assert all(item["split"] == "development" for item in development)
    assert all(item["split"] == "evaluation" for item in evaluation)
