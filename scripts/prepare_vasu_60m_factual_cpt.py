"""Prepare and validate the isolated VASU-60M factual CPT data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vasu.data.preparation.reporting import sha256_file
from vasu.data.pretraining_mixture import (
    MIXTURE_ARTIFACT_FORMAT,
    TOKENIZED_FORMAT,
    FixedRecordTokenDataset,
    build_mixture_artifact,
    prepare_wikimedia_tokenized_split,
)


SOURCE = Path(
    "data/processed/pretrain/factual/wikimedia_pilot_release/documents.jsonl"
)
SOURCE_SHA256 = "6aa10d73669ca90ad20f867f14a6368d2191b38094f02aa1679ebaf122962de7"
TOKENIZER = Path("assets/tokenizer.json")
TOKENIZER_SHA256 = "04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a"
TRAIN_BIN = Path(
    "data/processed/pretrain/factual/wikimedia_release_6aa10d73_train.bin"
)
VALIDATION_BIN = Path(
    "data/processed/pretrain/factual/wikimedia_release_6aa10d73_validation.bin"
)
TOKENIZED_MANIFEST = Path(
    "data/manifests/factual/wikimedia_release_6aa10d73_tokenized.json"
)
MIXTURE_CONFIG = Path(
    "configs/data/mixtures/vasu_60m_fineweb_wikimedia_85_15.json"
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def validate_outputs() -> dict[str, object]:
    if sha256_file(SOURCE) != SOURCE_SHA256:
        raise ValueError("approved Wikimedia release changed")
    if sha256_file(TOKENIZER) != TOKENIZER_SHA256:
        raise ValueError("configured tokenizer changed")
    tokenized = _load(TOKENIZED_MANIFEST)
    if tokenized.get("format_version") != TOKENIZED_FORMAT:
        raise ValueError("unsupported Wikimedia tokenization manifest")
    if tokenized.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("tokenization manifest source hash mismatch")
    if tokenized.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("tokenization manifest tokenizer hash mismatch")
    train = tokenized["train"]
    validation = tokenized["validation"]
    if set(train["parent_ids"]).intersection(validation["parent_ids"]):
        raise ValueError("Wikimedia parent leakage detected")
    for split, path in ((train, TRAIN_BIN), (validation, VALIDATION_BIN)):
        if sha256_file(path) != split["sha256"]:
            raise ValueError(f"tokenized split hash mismatch: {path}")
        tokens = np.memmap(path, dtype=np.uint16, mode="r")
        if len(tokens) != split["token_count"]:
            raise ValueError(f"tokenized split count mismatch: {path}")
        if not len(tokens) or int(tokens.max()) >= 32_000:
            raise ValueError(f"invalid token IDs in {path}")

    mixture_config = _load(MIXTURE_CONFIG)
    metadata_path = Path(mixture_config["output"]["metadata"])
    metadata = _load(metadata_path)
    if metadata.get("format_version") != MIXTURE_ARTIFACT_FORMAT:
        raise ValueError("unsupported mixture artifact metadata")
    mixture_path = Path(metadata["output_path"])
    schedule_path = Path(metadata["selection_schedule_path"])
    if sha256_file(mixture_path) != metadata["output_sha256"]:
        raise ValueError("mixture artifact hash mismatch")
    if sha256_file(schedule_path) != metadata["selection_schedule_sha256"]:
        raise ValueError("selection schedule hash mismatch")
    dataset = FixedRecordTokenDataset(
        mixture_path,
        int(metadata["sequence_length"]),
        expected_sha256=metadata["output_sha256"],
    )
    if len(dataset) != metadata["record_count"]:
        raise ValueError("mixture record count mismatch")
    x, y = dataset[0]
    if x.shape != y.shape or x.shape != (metadata["sequence_length"],):
        raise ValueError("mixture sample shape mismatch")
    return {
        "wikimedia_train_parents": train["parent_count"],
        "wikimedia_validation_parents": validation["parent_count"],
        "wikimedia_train_tokens": train["token_count"],
        "wikimedia_validation_tokens": validation["token_count"],
        "parent_overlap": 0,
        "mixture_records": len(dataset),
        "mixture_output_sha256": metadata["output_sha256"],
        "selection_schedule_sha256": metadata["selection_schedule_sha256"],
        "actual_supervised_token_budget": metadata[
            "actual_supervised_token_budget"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--tokenize-wikimedia", action="store_true")
    actions.add_argument("--build-mixture", action="store_true")
    actions.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.tokenize_wikimedia:
        result = prepare_wikimedia_tokenized_split(
            source_path=SOURCE,
            expected_source_sha256=SOURCE_SHA256,
            tokenizer_path=TOKENIZER,
            expected_tokenizer_sha256=TOKENIZER_SHA256,
            output_train_path=TRAIN_BIN,
            output_validation_path=VALIDATION_BIN,
            output_manifest_path=TOKENIZED_MANIFEST,
        )
    elif args.build_mixture:
        result = build_mixture_artifact(MIXTURE_CONFIG)
    else:
        result = validate_outputs()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
