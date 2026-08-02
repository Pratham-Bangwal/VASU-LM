"""Deterministic self-curated likelihood inventories for VASU-140M eval v2."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from tokenizers import Tokenizer

from evaluation.framework.vasu_140m_base_v2 import (
    INTERFACES,
    TOKENIZER_SHA256,
    canonical_json,
    sha256_file,
)
from evaluation.framework.vasu_140m_base_v2_inventory import (
    CONTAMINATION_RECORD_SCHEMA_ID,
    INVENTORY_SCHEMA_ID,
    PAYLOAD_RECORD_SCHEMA_ID,
    PROVENANCE_RECORD_SCHEMA_ID,
    content_text_sha256,
    inventory_identity,
    normalized_text_sha256,
    normalized_word_count,
    provenance_identity,
    validate_contamination_record,
    validate_inventory_manifest,
    validate_inventory_manifest_files,
    validate_payload_record,
    validate_provenance_record,
)
from evaluation.framework.vasu_140m_base_v2_tasks import (
    TASK_SCHEMA_ID,
    task_identity,
    validate_task,
)


SELECTION_SEED = 140021
ITEMS_PER_SPLIT = 512
CONTEXT_TOKENS = 256
MAX_RECORD_TOKENS = 513
MIN_DOCUMENT_TOKENS = 128
NGRAM_WORDS = 8
SCORER_PATH = "evaluation/framework/vasu_140m_base_v2_tasks.py"


def selection_rank(source_id: str, parent_document_id: str) -> str:
    material = f"{SELECTION_SEED}:{source_id}:{parent_document_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def reserve_documents(
    documents: Iterable[Mapping[str, str]], source_id: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Reserve the lowest SHA-ranked whole documents without loading a corpus."""

    limit = ITEMS_PER_SPLIT * 2
    heap: list[tuple[int, str, dict[str, str]]] = []
    seen: set[str] = set()
    for document in documents:
        parent_id = str(document["parent_document_id"])
        if parent_id in seen:
            raise ValueError("source iterator contains duplicate parent documents")
        seen.add(parent_id)
        rank = selection_rank(source_id, parent_id)
        entry = (-int(rank, 16), parent_id, dict(document))
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    if len(heap) < limit:
        raise ValueError(f"{source_id} has fewer than {limit} eligible documents")
    selected = sorted(
        (entry[2] for entry in heap),
        key=lambda item: selection_rank(source_id, item["parent_document_id"]),
    )
    return selected[:ITEMS_PER_SPLIT], selected[ITEMS_PER_SPLIT:]


def iter_fineweb_documents(path: Path) -> Iterable[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            text = record.get("text")
            parent_id = record.get("historical_source_id")
            if not isinstance(text, str) or not text.strip() or not parent_id:
                raise ValueError(f"invalid FineWeb record at line {line_number}")
            if len(text) < 700:
                continue
            yield {
                "parent_document_id": str(parent_id),
                "text": text,
                "source_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
                "source_revision": str(record["pinned_revision"]),
                "citation": (
                    f"{record['stable_row_reference']}; {record['provider_shard']}"
                ),
                "retrieved_at": str(record["retrieval_timestamp"]),
            }


def iter_wikimedia_documents(path: Path, retrieved_at: str) -> Iterable[dict[str, str]]:
    parents: dict[str, list[dict[str, object]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            parent_id = record.get("parent_document_id")
            if not parent_id or not isinstance(record.get("cleaned_text"), str):
                raise ValueError(f"invalid Wikimedia record at line {line_number}")
            parents[str(parent_id)].append(record)
    for parent_id, chunks in parents.items():
        eligible = [
            item for item in chunks if int(item.get("token_count", 0)) >= MIN_DOCUMENT_TOKENS
        ]
        if not eligible:
            continue
        chosen = min(
            eligible,
            key=lambda item: hashlib.sha256(
                f"{SELECTION_SEED}:{parent_id}:{item['chunk_id']}".encode()
            ).hexdigest(),
        )
        yield {
            "parent_document_id": parent_id,
            "text": str(chosen["cleaned_text"]),
            "source_url": str(chosen["source_url"]),
            "source_revision": str(chosen["source_revision"]),
            "citation": f"{chosen['title']}; chunk {chosen['chunk_id']}",
            "retrieved_at": retrieved_at,
        }


def likelihood_content(
    tokenizer: Tokenizer, text: str, rank: str
) -> tuple[str, str, int]:
    token_ids = tokenizer.encode(text).ids
    if len(token_ids) < MIN_DOCUMENT_TOKENS:
        raise ValueError("selected document is too short for likelihood scoring")
    window_size = min(MAX_RECORD_TOKENS, len(token_ids))
    maximum_start = len(token_ids) - window_size
    start = int(hashlib.sha256(f"window:{rank}".encode()).hexdigest(), 16) % (
        maximum_start + 1
    )
    window = token_ids[start : start + window_size]
    context_width = min(CONTEXT_TOKENS, len(window) - 1)
    context = tokenizer.decode(window[:context_width])
    target_ids = window[context_width:]
    while target_ids:
        target = tokenizer.decode(target_ids)
        combined_count = len(tokenizer.encode(context + target).ids)
        target_count = len(tokenizer.encode(target).ids)
        if target.strip() and target_count and combined_count <= MAX_RECORD_TOKENS:
            return context, target, target_count
        target_ids = target_ids[:-1]
    raise ValueError("could not construct a bounded non-empty likelihood target")


def _exact_commitments(values: Sequence[str]) -> list[dict[str, object]]:
    unique = {
        normalized_text_sha256(value): normalized_word_count(value) for value in values
    }
    return [
        {"sha256": digest, "word_count": unique[digest]}
        for digest in sorted(unique)
    ]


def _ngram_hashes(values: Sequence[str]) -> list[str]:
    identities: set[str] = set()
    for value in values:
        words = value.strip().split()
        for index in range(max(0, len(words) - NGRAM_WORDS + 1)):
            identities.add(
                normalized_text_sha256(" ".join(words[index : index + NGRAM_WORDS]))
            )
    return sorted(identities)


def build_records(
    *,
    tokenizer: Tokenizer,
    source_id: str,
    source_name: str,
    license_name: str,
    license_url: str,
    split: str,
    documents: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payloads: list[dict[str, object]] = []
    provenance_records: list[dict[str, object]] = []
    contamination_records: list[dict[str, object]] = []
    for index, document in enumerate(documents):
        parent_id = str(document["parent_document_id"])
        rank = selection_rank(source_id, parent_id)
        item_id = f"likelihood-{source_id}-{split}-{index:04d}"
        context, target, target_count = likelihood_content(
            tokenizer, str(document["text"]), rank
        )
        text_sha = content_text_sha256(context + target)
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": "likelihood",
            "split": split,
            "strata": {"source": source_id},
            "input": {"text_sha256": text_sha, "target_token_count": target_count},
            "scoring": {"kind": "token_log_likelihood"},
        }
        validate_task(task)
        payload = {
            "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
            "task": task,
            "content": {
                "context": context,
                "target": target,
                "text_sha256": text_sha,
                "target_sha256": content_text_sha256(target),
            },
        }
        if split == "development":
            validate_payload_record(payload, "likelihood")
        provenance = {
            "schema_id": PROVENANCE_RECORD_SCHEMA_ID,
            "item_id": item_id,
            "source_name": source_name,
            "source_url": document["source_url"],
            "license_name": license_name,
            "license_url": license_url,
            "source_revision": document["source_revision"],
            "citation": document["citation"],
            "parent_document_id": parent_id,
            "retrieved_at": document["retrieved_at"],
            "human_authored": True,
        }
        validate_provenance_record(provenance)
        semantic_material = f"{source_id}:{parent_id}:{text_sha}"
        contamination = {
            "schema_id": CONTAMINATION_RECORD_SCHEMA_ID,
            "item_id": item_id,
            "parent_document_id": parent_id,
            "prompt_exact_commitments": _exact_commitments([context]),
            "answer_exact_commitments": _exact_commitments([target]),
            "ngram_words": NGRAM_WORDS,
            "ngram_sha256s": _ngram_hashes([context, target]),
            "semantic_fingerprint": {
                "method": "source-parent-and-content-sha256-v1",
                "value": hashlib.sha256(semantic_material.encode()).hexdigest(),
            },
        }
        validate_contamination_record(contamination)
        payloads.append(payload)
        provenance_records.append(provenance)
        contamination_records.append(contamination)
    return payloads, provenance_records, contamination_records


def _jsonl(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json(record) + b"\n" for record in records)


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _binding(path: Path, relative: str, record_count: int) -> dict[str, object]:
    return {
        "path": relative.replace("\\", "/"),
        "sha256": sha256_file(path),
        "record_count": record_count,
        "byte_count": path.stat().st_size,
    }


def build_inventory(
    *,
    repository_root: Path,
    relative_output: str,
    suite_id: str,
    repository_commit: str,
    source_id: str,
    split: str,
    recipient: str,
    payloads: Sequence[Mapping[str, object]],
    provenance: Sequence[Mapping[str, object]],
    contamination: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if split not in {"development", "held_out"}:
        raise ValueError("split is unsupported")
    if not (len(payloads) == len(provenance) == len(contamination) == ITEMS_PER_SPLIT):
        raise ValueError("likelihood inventory must contain exactly 512 records")
    root = repository_root.resolve()
    destination = (root / relative_output).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("output path escapes repository")
    if destination.exists():
        raise FileExistsError("inventory output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError("stale inventory staging directory exists")
    staging.mkdir()
    try:
        payload_name = "payload.jsonl" if split == "development" else "payload.jsonl.age"
        payload_bytes = _jsonl(payloads)
        if split == "development":
            _write_new(staging / payload_name, payload_bytes)
        else:
            age = shutil.which("age")
            if age is None:
                raise RuntimeError("age executable is unavailable")
            with tempfile.TemporaryDirectory(prefix="vasu-heldout-") as temporary:
                plaintext = Path(temporary) / "payload.jsonl"
                _write_new(plaintext, payload_bytes)
                result = subprocess.run(
                    [age, "--recipient", recipient, "--output", str(staging / payload_name), str(plaintext)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"age encryption failed: {result.stderr.strip()}")
        _write_new(staging / "provenance.jsonl", _jsonl(provenance))
        _write_new(staging / "contamination.jsonl", _jsonl(contamination))
        relative = relative_output.replace("\\", "/")
        scorer = root / SCORER_PATH
        manifest: dict[str, object] = {
            "schema_id": INVENTORY_SCHEMA_ID,
            "inventory_id": f"{suite_id}-{source_id}-{split}",
            "suite_id": suite_id,
            "repository_commit": repository_commit,
            "dimension": "likelihood",
            "split": split,
            "interface": INTERFACES["likelihood"],
            "tokenizer_sha256": TOKENIZER_SHA256,
            "payload": _binding(staging / payload_name, f"{relative}/{payload_name}", len(payloads)),
            "provenance_index": _binding(staging / "provenance.jsonl", f"{relative}/provenance.jsonl", len(provenance)),
            "contamination_index": _binding(staging / "contamination.jsonl", f"{relative}/contamination.jsonl", len(contamination)),
            "scorer": {"path": SCORER_PATH, "sha256": sha256_file(scorer)},
            "generation_profile_ids": [],
            "item_commitments": [
                {
                    "item_id": payload["task"]["item_id"],
                    "task_sha256": task_identity(payload["task"]),
                    "provenance_sha256": provenance_identity(provenance_record),
                }
                for payload, provenance_record in zip(payloads, provenance, strict=True)
            ],
            "access": (
                {
                    "state": "available",
                    "payload_format": "jsonl",
                    "encryption_algorithm": "none",
                    "recipient_fingerprint": None,
                    "opening_authorized": False,
                }
                if split == "development"
                else {
                    "state": "sealed",
                    "payload_format": "encrypted_jsonl",
                    "encryption_algorithm": "age-x25519",
                    "recipient_fingerprint": hashlib.sha256(recipient.encode()).hexdigest(),
                    "opening_authorized": False,
                }
            ),
            "fixture_only": True,
            "production_suite_frozen": False,
            "evaluation_run_authorized": False,
            "training_authorized": False,
        }
        manifest["inventory_sha256"] = inventory_identity(manifest)
        validate_inventory_manifest(manifest)
        _write_new(
            staging / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n",
        )
        os.replace(staging, destination)
        validate_inventory_manifest_files(manifest, root)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
