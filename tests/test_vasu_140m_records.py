from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from scripts.smoke_vasu_140m_record_spec import run
from vasu.data.vasu_140m_records import (
    CONTEXT_LENGTH,
    EOS_TOKEN_ID,
    FROZEN_FIXTURE_REPORT_SHA256,
    LogicalExample,
    RECORD_WIDTH,
    SPECIFICATION_SHA256,
    build_fixture_report,
    compile_text_example,
    pack_split,
    sha256_json,
    shifted_training_view,
    specification,
    validate_frozen_fixture_report,
    validate_logical_example,
    validate_fixture_report,
    validate_split_isolation,
)


TOKENIZER_PATH = Path("assets/tokenizer.json")


def _example(example_id: str, split: str, marker: int) -> LogicalExample:
    return LogicalExample(
        example_id=example_id,
        split=split,
        semantic_sha256=f"{marker:064x}",
        token_ids=(10 + marker, 20 + marker, EOS_TOKEN_ID),
        stored_mask=(0, 1, 1),
        target_start=1,
    )


def _splits() -> dict[str, list[LogicalExample]]:
    return {
        "train": [_example("train-1", "train", 1), _example("train-2", "train", 2)],
        "development": [_example("dev-1", "development", 3)],
        "evaluation": [_example("eval-1", "evaluation", 4)],
    }


def test_frozen_specification_identity_and_width() -> None:
    assert CONTEXT_LENGTH == 512
    assert RECORD_WIDTH == 513
    assert len(SPECIFICATION_SHA256) == 64
    assert specification()["training_authorized"] is False
    assert specification()["production_release_created"] is False


def test_real_tokenizer_boundary_and_fixture_rebuild_are_deterministic() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    example = compile_text_example(
        tokenizer=tokenizer,
        example_id="boundary",
        split="train",
        prompt="User: Add one and one.\nAssistant:\n",
        response="Answer: 2",
    )
    assert example.stored_mask[: example.target_start] == (0,) * example.target_start
    assert all(example.stored_mask[example.target_start :])
    rebuilt = run(TOKENIZER_PATH)
    assert rebuilt == run(TOKENIZER_PATH)
    frozen = json.loads(
        Path("evaluation/fixtures/vasu_140m_513_record_spec_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert rebuilt == frozen
    assert rebuilt["report_sha256"] == FROZEN_FIXTURE_REPORT_SHA256
    validate_frozen_fixture_report(rebuilt)


def test_shifted_mask_hides_prompt_pad_and_cross_example_transition() -> None:
    records = pack_split(_splits()["train"], split="train")
    assert len(records) == 1
    record = records[0]
    assert record.tokens.dtype == np.uint16
    assert record.stored_mask.dtype == np.uint8
    assert record.tokens.shape == record.stored_mask.shape == (513,)
    second = record.spans[1]
    shifted_mask = record.stored_mask[1:]
    assert shifted_mask[second.start - 1] == 0
    assert np.all(record.stored_mask[record.used_token_count :] == 0)
    assert np.all(record.tokens[record.used_token_count :] == 0)


def test_shifted_views_are_record_local_and_never_join_records() -> None:
    first = LogicalExample(
        example_id="wide-1",
        split="train",
        semantic_sha256="b" * 64,
        token_ids=(10,) * 299 + (EOS_TOKEN_ID,),
        stored_mask=(0,) + (1,) * 299,
        target_start=1,
    )
    second = replace(
        first,
        example_id="wide-2",
        semantic_sha256="c" * 64,
        token_ids=(11,) * 299 + (EOS_TOKEN_ID,),
    )
    records = pack_split([first, second], split="train")
    assert len(records) == 2
    views = [shifted_training_view(record) for record in records]
    assert all(view.input_ids.shape == (512,) for view in views)
    assert all(view.target_ids.shape == (512,) for view in views)
    assert all(view.loss_mask.shape == (512,) for view in views)
    assert views[0].target_ids[-1] == 0
    assert views[1].input_ids[0] == 11
    assert not np.shares_memory(views[0].target_ids, records[1].tokens)


def test_split_isolation_rejects_id_and_semantic_leakage() -> None:
    splits = _splits()
    splits["evaluation"][0] = replace(splits["evaluation"][0], example_id="train-1")
    with pytest.raises(ValueError, match="duplicate example_id"):
        validate_split_isolation(splits)
    splits = _splits()
    splits["evaluation"][0] = replace(
        splits["evaluation"][0],
        semantic_sha256=splits["train"][0].semantic_sha256,
    )
    with pytest.raises(ValueError, match="semantic split leakage"):
        validate_split_isolation(splits)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"token_ids": (10, 20), "stored_mask": (0, 1)}, "terminal EOS"),
        ({"token_ids": (0, 20, 3)}, "contains PAD"),
        ({"stored_mask": (1, 1, 1)}, "prompt is supervised"),
        ({"stored_mask": (0, 1, 0)}, "response or EOS is unsupervised"),
        ({"token_ids": (32_000, 20, 3)}, "out-of-vocabulary"),
    ],
)
def test_logical_example_validation_fails_closed(
    mutation: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_logical_example(replace(_example("x", "train", 5), **mutation))


def test_exact_width_allowed_and_overlength_rejected() -> None:
    exact = LogicalExample(
        example_id="exact",
        split="train",
        semantic_sha256="a" * 64,
        token_ids=(10,) * 512 + (EOS_TOKEN_ID,),
        stored_mask=(0,) + (1,) * 512,
        target_start=1,
    )
    assert pack_split([exact], split="train")[0].used_token_count == 513
    with pytest.raises(ValueError, match="exceeds"):
        validate_logical_example(
            replace(
                exact,
                token_ids=(10,) * 513 + (EOS_TOKEN_ID,),
                stored_mask=(0,) + (1,) * 513,
            )
        )


def test_fixture_report_has_no_release_or_authorization_semantics() -> None:
    report = build_fixture_report(_splits())
    assert report["fixture_only"] is True
    assert report["production_release_created"] is False
    assert report["training_authorized"] is False
    assert all(report["checks"].values())
    assert report["family_id"] == "vasu_140m_v1"
    assert report["record_width"] == 513
    assert len(report["model_config_sha256"]) == 64
    assert len(report["tokenizer_sha256"]) == 64
    assert len(report["logical_input_sha256"]) == 64
    validate_fixture_report(report)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("family_id",), "vasu_60m_v1", "family_id"),
        (("tokenizer_sha256",), "0" * 64, "tokenizer_sha256"),
        (("record_width",), 257, "record_width"),
        (("checks", "cross_record_isolation"), False, "checks"),
        (("splits", "train", "token_sha256"), "0" * 64, "hash mismatch"),
        (("training_authorized",), True, "training_authorized"),
    ],
)
def test_fixture_report_validation_rejects_identity_tampering(
    path: tuple[str, ...], value: object, message: str
) -> None:
    report = deepcopy(build_fixture_report(_splits()))
    target = report
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_fixture_report(report)


def test_frozen_fixture_report_rejects_self_consistent_rehashed_mutation() -> None:
    report = deepcopy(run(TOKENIZER_PATH))
    report["logical_input_sha256"] = "0" * 64
    body = dict(report)
    del body["report_sha256"]
    report["report_sha256"] = sha256_json(body)

    validate_fixture_report(report)
    with pytest.raises(ValueError, match="frozen fixture identity"):
        validate_frozen_fixture_report(report)
