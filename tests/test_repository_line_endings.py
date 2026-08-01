from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LF_BOUND_FILES = (
    ".gitattributes",
    "assets/tokenizer.json",
    "evaluation/framework/vasu_140m_base_v2_tasks.py",
    "evaluation/framework/vasu_140m_base_v2_statistics.py",
    "evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json",
    "scripts/smoke_vasu_140m_base_v2_scoring.py",
    "tests/test_vasu_140m_base_v2_scoring.py",
)
BINARY_FILES = (
    "data/processed/fineweb_edu/train.bin",
    "checkpoints/vasu_60m/milestones/fineweb_step_200000.pt",
)


def _attributes(path: str) -> dict[str, str]:
    result = subprocess.run(
        ["git", "check-attr", "text", "eol", "--", path],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    attributes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        _, name, value = line.split(": ", 2)
        attributes[name] = value
    return attributes


def test_hash_bound_text_files_are_forced_to_lf() -> None:
    for path in LF_BOUND_FILES:
        assert _attributes(path) == {"text": "auto", "eol": "lf"}


def test_binary_research_artifacts_disable_text_normalization() -> None:
    for path in BINARY_FILES:
        attributes = _attributes(path)
        assert attributes["text"] == "unset"


def test_checked_out_hash_bound_files_contain_no_crlf() -> None:
    for relative in LF_BOUND_FILES:
        assert b"\r\n" not in (ROOT / relative).read_bytes()
