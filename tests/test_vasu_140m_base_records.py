from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vasu.data.vasu_140m_base_records import (
    BaseChunkStreamValidator,
    MASKING_CONTRACT,
    build_base_fixture_report,
    compile_base_text_chunk,
    iter_pack_base_chunks,
    pack_base_splits,
    validate_base_fixture_report,
    validate_base_split_isolation,
    validate_base_text_chunk,
)
from vasu.data.vasu_140m_records import (
    EOS_TOKEN_ID,
    RECORD_WIDTH,
    pack_split,
    shifted_training_view,
)


ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [10 + (ord(character) % 200) for character in text]


class FixedTokenizer:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens

    def encode(self, text: str) -> list[int]:
        return self.tokens


def chunk(
    *,
    split: str,
    chunk_id: str,
    text: str,
    source_id: str = "source-a",
    revision: str = "revision-1",
    document_id: str | None = None,
    chunk_index: int = 0,
):
    return compile_base_text_chunk(
        tokenizer=CharacterTokenizer(),
        source_id=source_id,
        source_revision=revision,
        document_id=document_id or f"document-{chunk_id}",
        document_sha256=(chunk_id.encode().hex() + "0" * 64)[:64],
        transformation_id="normalized-chunk-v1",
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        split=split,
        text=text,
    )


def splits():
    return {
        "train": [chunk(split="train", chunk_id="train-1", text="alpha text")],
        "development": [
            chunk(split="development", chunk_id="dev-1", text="beta text")
        ],
        "evaluation": [
            chunk(split="evaluation", chunk_id="eval-1", text="gamma text")
        ],
    }


def test_base_chunk_masks_only_boundary_and_appends_supervised_eos() -> None:
    value = chunk(split="train", chunk_id="train-1", text="base text")
    assert value.token_ids[-1] == EOS_TOKEN_ID
    assert value.stored_mask[0] == 0
    assert all(mask == 1 for mask in value.stored_mask[1:])
    assert sum(value.stored_mask) == len(value.token_ids) - 1
    validate_base_text_chunk(value)


@pytest.mark.parametrize("reserved_id", [0, 1, 2, 3])
def test_reserved_or_unknown_content_tokens_fail_closed(reserved_id: int) -> None:
    with pytest.raises(ValueError, match="reserved/unknown"):
        compile_base_text_chunk(
            tokenizer=FixedTokenizer([10, reserved_id, 11]),
            source_id="source",
            source_revision="revision",
            document_id="document",
            document_sha256="a" * 64,
            transformation_id="transform",
            chunk_id="chunk",
            chunk_index=0,
            split="train",
            text="content",
        )


def test_overwidth_and_blank_chunks_fail_closed() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        compile_base_text_chunk(
            tokenizer=FixedTokenizer([10] * RECORD_WIDTH),
            source_id="source",
            source_revision="revision",
            document_id="document",
            document_sha256="a" * 64,
            transformation_id="transform",
            chunk_id="chunk",
            chunk_index=0,
            split="train",
            text="content",
        )
    with pytest.raises(ValueError, match="text must be"):
        chunk(split="train", chunk_id="blank", text="  ")


def test_split_isolation_rejects_parent_crossing_and_duplicate_text() -> None:
    value = splits()
    value["evaluation"][0] = replace(
        value["evaluation"][0],
        source_id=value["train"][0].source_id,
        source_revision=value["train"][0].source_revision,
        document_id=value["train"][0].document_id,
        document_sha256=value["train"][0].document_sha256,
    )
    with pytest.raises(ValueError, match="parent document crosses splits"):
        validate_base_split_isolation(value)

    value = splits()
    value["evaluation"][0] = replace(
        value["evaluation"][0], text_sha256=value["train"][0].text_sha256
    )
    with pytest.raises(ValueError, match="duplicate chunk text"):
        validate_base_split_isolation(value)


def test_source_revision_and_parent_chunk_index_are_unique() -> None:
    value = splits()
    value["train"].append(
        chunk(
            split="train",
            chunk_id="train-2",
            text="delta text",
            source_id="source-a",
            revision="revision-2",
        )
    )
    with pytest.raises(ValueError, match="source revision changed"):
        validate_base_split_isolation(value)

    value = splits()
    first = value["train"][0]
    value["train"].append(
        chunk(
            split="train",
            chunk_id="train-2",
            text="epsilon text",
            document_id=first.document_id,
            chunk_index=0,
        )
    )
    with pytest.raises(ValueError, match="duplicate chunk_index"):
        validate_base_split_isolation(value)


def test_packing_masks_cross_chunk_and_pad_targets() -> None:
    value = splits()
    value["train"].append(
        chunk(split="train", chunk_id="train-2", text="delta text")
    )
    packed = pack_base_splits(value)
    record = packed["train"][0]
    assert len(record.spans) == 2
    second = record.spans[1]
    assert record.stored_mask[second.start] == 0
    training_view = shifted_training_view(record)
    assert training_view.loss_mask[second.start - 1] == 0
    assert not training_view.loss_mask[record.used_token_count - 1 :].any()


def test_streaming_packer_matches_frozen_packer_without_reiterating() -> None:
    chunks = [
        chunk(split="train", chunk_id="train-1", text="alpha text"),
        chunk(split="train", chunk_id="train-2", text="delta text"),
    ]
    iterations = 0

    def one_pass():
        nonlocal iterations
        iterations += 1
        yield from chunks

    streamed = list(iter_pack_base_chunks(one_pass(), split="train"))
    expected = pack_split(
        [item.logical_example() for item in chunks], split="train"
    )
    assert iterations == 1
    assert len(streamed) == len(expected)
    for actual, reference in zip(streamed, expected, strict=True):
        assert actual.tokens.tobytes() == reference.tokens.tobytes()
        assert actual.stored_mask.tobytes() == reference.stored_mask.tobytes()
        assert actual.spans == reference.spans


def test_stream_validator_tracks_global_isolation_incrementally() -> None:
    validator = BaseChunkStreamValidator()
    for rows in splits().values():
        for item in rows:
            validator.consume(item)
    assert validator.finish() == {"train": 1, "development": 1, "evaluation": 1}

    validator = BaseChunkStreamValidator()
    validator.consume(chunk(split="train", chunk_id="train-1", text="alpha"))
    with pytest.raises(ValueError, match="empty splits"):
        validator.finish()


def test_streaming_packer_rejects_empty_and_wrong_split_streams() -> None:
    with pytest.raises(ValueError, match="at least one base chunk"):
        list(iter_pack_base_chunks(iter(()), split="train"))
    value = chunk(split="development", chunk_id="dev-1", text="alpha")
    with pytest.raises(ValueError, match="cannot be packed"):
        list(iter_pack_base_chunks(iter([value]), split="train"))


def test_fixture_report_is_deterministic_complete_and_non_authorizing() -> None:
    first = build_base_fixture_report(splits())
    second = build_base_fixture_report(splits())
    assert first == second
    assert first["masking_contract"] == MASKING_CONTRACT
    assert first["source_discovered"] is False
    assert first["data_acquired"] is False
    assert first["production_release_created"] is False
    assert first["training_authorized"] is False
    for split in ("train", "development", "evaluation"):
        report = first["splits"][split]
        assert report["masked_chunk_boundary_count"] == report["chunk_count"]
        assert report["supervised_target_count"] == report["real_token_count"] - report["chunk_count"]


def test_report_unknown_fields_authorization_and_mutation_fail_closed() -> None:
    value = build_base_fixture_report(splits())
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_base_fixture_report(value)
    value = build_base_fixture_report(splits())
    value["training_authorized"] = True
    with pytest.raises(ValueError, match="training_authorized mismatch"):
        validate_base_fixture_report(value)
    value = build_base_fixture_report(splits())
    value["splits"]["train"]["chunk_count"] += 1
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_base_fixture_report(value)


def test_frozen_smoke_fixture_reproduces_exactly() -> None:
    from scripts.smoke_vasu_140m_base_records import run

    expected = json.loads(
        (ROOT / "evaluation/fixtures/vasu_140m_base_text_record_fixture_v1.json")
        .read_text(encoding="utf-8")
    )
    assert run() == expected
