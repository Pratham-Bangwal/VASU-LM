from __future__ import annotations

import gzip
import json
from pathlib import Path

from tokenizers import Tokenizer

from evaluation.framework.vasu_140m_likelihood_inventory import (
    ITEMS_PER_SPLIT,
    MAX_RECORD_TOKENS,
    likelihood_content,
    iter_fineweb_documents,
    reserve_documents,
    selection_rank,
)


def _documents(count: int) -> list[dict[str, str]]:
    return [
        {"parent_document_id": f"doc-{index:04d}", "text": "content"}
        for index in range(count)
    ]


def test_reservation_is_deterministic_disjoint_and_sha_ranked() -> None:
    source_id = "source-a"
    development, held_out = reserve_documents(_documents(1200), source_id)
    assert len(development) == ITEMS_PER_SPLIT
    assert len(held_out) == ITEMS_PER_SPLIT
    development_ids = {item["parent_document_id"] for item in development}
    held_out_ids = {item["parent_document_id"] for item in held_out}
    assert not development_ids & held_out_ids
    combined = [*development, *held_out]
    assert combined == sorted(
        combined,
        key=lambda item: selection_rank(source_id, item["parent_document_id"]),
    )
    assert reserve_documents(reversed(_documents(1200)), source_id) == (
        development,
        held_out,
    )


def test_reservation_fails_when_source_is_too_small() -> None:
    try:
        reserve_documents(_documents(ITEMS_PER_SPLIT * 2 - 1), "small")
    except ValueError as error:
        assert "fewer than 1024" in str(error)
    else:
        raise AssertionError("undersized source was accepted")


def test_reservation_rejects_duplicate_parent_ids() -> None:
    documents = _documents(ITEMS_PER_SPLIT * 2)
    documents[-1] = dict(documents[0])
    try:
        reserve_documents(documents, "duplicate")
    except ValueError as error:
        assert "duplicate parent" in str(error)
    else:
        raise AssertionError("duplicate parent was accepted")


def test_likelihood_content_is_nonempty_and_token_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    tokenizer = Tokenizer.from_file(str(root / "assets/tokenizer.json"))
    text = " ".join(f"token-{index}" for index in range(1000))
    context, target, target_count = likelihood_content(tokenizer, text, "a" * 64)
    assert context
    assert target
    assert target_count == len(tokenizer.encode(target).ids)
    assert len(tokenizer.encode(context + target).ids) <= MAX_RECORD_TOKENS


def test_fineweb_iterator_excludes_documents_below_token_floor(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl.gz"
    records = []
    for index, word_count in enumerate((127, 128)):
        records.append(
            {
                "historical_source_id": f"doc-{index}",
                "text": " ".join(["word"] * word_count),
                "pinned_revision": "revision",
                "stable_row_reference": f"row:{index}",
                "provider_shard": "shard",
                "retrieval_timestamp": "2026-08-02T00:00:00+00:00",
            }
        )
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    observed = list(iter_fineweb_documents(source))
    assert [item["parent_document_id"] for item in observed] == ["doc-1"]
