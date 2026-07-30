from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vasu.data.vasu_140m_fixture_release import (
    DECISION_SHA256,
    PLAN_SHA256,
    build_fixture_release,
    validate_fixture_release,
)
from vasu.data.vasu_140m_records import (
    EOS_TOKEN_ID,
    RECORD_WIDTH,
    compile_text_example,
)


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [10 + (ord(character) % 200) for character in text]


def fixture_splits() -> dict[str, list]:
    tokenizer = CharacterTokenizer()
    return {
        split: [
            compile_text_example(
                tokenizer=tokenizer,
                example_id=f"{split}-{index}",
                split=split,
                prompt=f"Prompt {split} {index}:",
                response=f" answer {index}",
            )
            for index in range(2)
        ]
        for split in ("train", "development", "evaluation")
    }


def test_fixture_release_is_deterministic_and_mask_aligned(tmp_path: Path) -> None:
    first = tmp_path / "first-fixture"
    second = tmp_path / "second-fixture"
    manifest_a = build_fixture_release(
        splits=fixture_splits(),
        output_dir=first,
        repository_root=tmp_path,
    )
    manifest_b = build_fixture_release(
        splits=fixture_splits(),
        output_dir=second,
        repository_root=tmp_path,
    )
    assert manifest_a == manifest_b
    assert manifest_a["fixture_only"] is True
    assert manifest_a["production_release_created"] is False
    assert manifest_a["training_authorized"] is False
    assert manifest_a["training_permitted"] is False
    assert json.loads((first / "manifest.json").read_text()) == manifest_a
    for split in ("train", "development", "evaluation"):
        tokens = np.fromfile(first / f"{split}.tokens.bin", dtype=np.uint16)
        mask = np.fromfile(first / f"{split}.mask.bin", dtype=np.uint8)
        assert tokens.size == mask.size
        assert tokens.size % RECORD_WIDTH == 0
        assert set(np.unique(mask)) <= {0, 1}
        assert np.all(mask[tokens == 0] == 0)
        assert np.all(mask[tokens == EOS_TOKEN_ID] == 1)
        assert np.array_equal(
            tokens,
            np.fromfile(second / f"{split}.tokens.bin", dtype=np.uint16),
        )
        assert np.array_equal(
            mask,
            np.fromfile(second / f"{split}.mask.bin", dtype=np.uint8),
        )


def test_fixture_release_rejects_overwrite_and_production_paths(
    tmp_path: Path,
) -> None:
    output = tmp_path / "safe-fixture"
    build_fixture_release(
        splits=fixture_splits(), output_dir=output, repository_root=tmp_path
    )
    with pytest.raises(FileExistsError, match="already exists"):
        build_fixture_release(
            splits=fixture_splits(), output_dir=output, repository_root=tmp_path
        )
    production = tmp_path / "data" / "processed" / "vasu_140m"
    production.mkdir(parents=True)
    with pytest.raises(ValueError, match="production release"):
        build_fixture_release(
            splits=fixture_splits(),
            output_dir=production / "instruction_seed_v1",
            repository_root=tmp_path,
        )
    with pytest.raises(ValueError, match="contain 'fixture'"):
        build_fixture_release(
            splits=fixture_splits(),
            output_dir=tmp_path / "ordinary-output",
            repository_root=tmp_path,
        )


def test_fixture_release_failure_leaves_no_partial_publication(
    tmp_path: Path,
) -> None:
    output = tmp_path / "failure-fixture"
    with pytest.raises(RuntimeError, match="injected"):
        build_fixture_release(
            splits=fixture_splits(),
            output_dir=output,
            repository_root=tmp_path,
            inject_failure_after_files=3,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".failure-fixture.staging-*"))


def test_fixture_release_detects_artifact_tampering(tmp_path: Path) -> None:
    output = tmp_path / "tamper-fixture"
    build_fixture_release(
        splits=fixture_splits(), output_dir=output, repository_root=tmp_path
    )
    with (output / "train.tokens.bin").open("ab") as stream:
        stream.write(b"\x00\x00")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_fixture_release(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("plan_sha256", "0" * 64, "plan identity"),
        ("decision_sha256", "f" * 64, "decision identity"),
    ],
)
def test_fixture_release_rejects_wrong_review_identity(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    arguments = {
        "splits": fixture_splits(),
        "output_dir": tmp_path / "identity-fixture",
        "repository_root": tmp_path,
        "plan_sha256": PLAN_SHA256,
        "decision_sha256": DECISION_SHA256,
    }
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        build_fixture_release(**arguments)


def test_fixture_release_hard_limits_fixture_size(tmp_path: Path) -> None:
    splits = fixture_splits()
    splits["train"] *= 16
    with pytest.raises(ValueError, match="example-count limit"):
        build_fixture_release(
            splits=splits,
            output_dir=tmp_path / "oversized-fixture",
            repository_root=tmp_path,
        )
