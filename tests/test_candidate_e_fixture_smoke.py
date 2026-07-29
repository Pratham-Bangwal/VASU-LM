from pathlib import Path

from scripts.smoke_candidate_e_release_fixture import run_fixture


def test_fixture_release_is_disposable_and_non_authorizing() -> None:
    report = run_fixture(Path("assets/tokenizer.json"))
    assert report == {
        "fixture_only": True,
        "training_authorized": False,
        "arms": ["final_answer", "verified_steps"],
        "source_split_counts": {"train": 12, "development": 4, "evaluation": 4},
    }
