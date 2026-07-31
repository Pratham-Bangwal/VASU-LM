"""Run prompt-free fixture qualification for VASU-140M base-text records."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.data.vasu_140m_base_records import (  # noqa: E402
    build_base_fixture_report,
    compile_base_text_chunk,
)
from vasu.data.vasu_140m_records import TOKENIZER_SHA256  # noqa: E402


FIXTURES = {
    "train": (
        (
            "fixture-web",
            "revision-1",
            "web-doc-1",
            "web-chunk-1",
            "A river crosses the quiet valley before reaching the sea.",
        ),
        (
            "fixture-wiki",
            "revision-1",
            "wiki-doc-1",
            "wiki-chunk-1",
            "Saturn is the sixth planet from the Sun and has a prominent ring system.",
        ),
    ),
    "development": (
        (
            "fixture-web",
            "revision-1",
            "web-doc-2",
            "web-chunk-2",
            "Careful measurements make scientific results easier to reproduce.",
        ),
    ),
    "evaluation": (
        (
            "fixture-wiki",
            "revision-1",
            "wiki-doc-2",
            "wiki-chunk-2",
            "A triangle has three sides and three interior angles.",
        ),
    ),
}


def run() -> dict[str, object]:
    tokenizer_path = REPOSITORY_ROOT / "assets/tokenizer.json"
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("tokenizer identity does not match the frozen contract")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    splits = {}
    for split, rows in FIXTURES.items():
        splits[split] = [
            compile_base_text_chunk(
                tokenizer=tokenizer,
                source_id=source_id,
                source_revision=revision,
                document_id=document_id,
                document_sha256=hashlib.sha256(
                    f"{source_id}:{revision}:{document_id}".encode("utf-8")
                ).hexdigest(),
                transformation_id="base-record-smoke-v1",
                chunk_id=chunk_id,
                chunk_index=0,
                split=split,
                text=text,
            )
            for source_id, revision, document_id, chunk_id, text in rows
        ]
    first = build_base_fixture_report(splits)
    if first != build_base_fixture_report(splits):
        raise RuntimeError("base-record fixture rebuild is not deterministic")
    return first


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
