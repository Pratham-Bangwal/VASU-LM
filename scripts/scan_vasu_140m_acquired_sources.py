"""Run hash-only contamination scans on acquired VASU-140M candidate sources."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vasu.data.vasu_140m_streaming_contamination import scan_documents  # noqa: E402


EXCLUSIONS = ROOT / "configs/data/exclusions/vasu_140m_likelihood_self_curated_v1.json"
FINEWEB = ROOT / "data/interim/pretrain/fineweb_extension_recovered.jsonl.gz"
WIKIMEDIA = ROOT / "data/processed/pretrain/factual/wikimedia_likelihood_source_v1/documents.jsonl"
OUTPUTS = {
    "fineweb": ROOT / "evaluation/results/vasu_140m_fineweb_exact_contamination_v2_20260802.json",
    "wikimedia": ROOT / "evaluation/results/vasu_140m_wikimedia_exact_contamination_v2_20260802.json",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contamination_records() -> list[dict[str, object]]:
    paths = sorted(
        (ROOT / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1").glob(
            "*/contamination.jsonl"
        )
    ) + sorted(
        (ROOT / "evaluation/fixtures/vasu_140m_likelihood_self_curated_v1").glob(
            "*/*/contamination.jsonl"
        )
    )
    records = []
    for path in paths:
        records.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    return records


def _excluded(source_id: str) -> set[str]:
    registry = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    source = next(item for item in registry["sources"] if item["source_id"] == source_id)
    return set(source["excluded_parent_document_ids"])


def _fineweb_documents():
    with gzip.open(FINEWEB, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            parent_id = str(record["historical_source_id"])
            yield {"document_id": parent_id, "parent_document_id": parent_id, "text": record["text"]}


def _wikimedia_documents():
    with WIKIMEDIA.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            yield {
                "document_id": str(record["document_id"]),
                "parent_document_id": str(record["parent_document_id"]),
                "text": record["cleaned_text"],
            }


def run(source: str) -> dict[str, object]:
    output = OUTPUTS[source]
    if output.exists():
        raise FileExistsError(output)
    if source == "fineweb":
        source_id = "fineweb_edu_extension_2025_26"
        artifact = FINEWEB
        documents = _fineweb_documents()
    else:
        source_id = "wikipedia_en_20231101"
        artifact = WIKIMEDIA
        documents = _wikimedia_documents()
    report = scan_documents(
        source_id=source_id,
        documents=documents,
        excluded_parent_ids=_excluded(source_id),
        contamination_records=_contamination_records(),
        source_artifact_sha256=_sha(artifact),
    )
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(OUTPUTS), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source), indent=2, sort_keys=True))
