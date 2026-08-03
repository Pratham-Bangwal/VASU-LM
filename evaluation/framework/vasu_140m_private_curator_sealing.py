"""Seal validated private curator inputs into non-authorizing held-out bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

from evaluation.framework.vasu_140m_base_v2 import (
    INTERFACES,
    RESULT_MODES,
    TOKENIZER_SHA256,
    canonical_json,
    sha256_file,
)
from evaluation.framework.vasu_140m_base_v2_inventory import (
    INVENTORY_SCHEMA_ID,
    inventory_identity,
    provenance_identity,
    validate_inventory_manifest,
    validate_inventory_manifest_files,
)
from evaluation.framework.vasu_140m_base_v2_inventory_builder import (
    AUTHORING_SCHEMA_ID,
    build_item_records,
)
from evaluation.framework.vasu_140m_base_v2_tasks import task_identity, validate_task
from evaluation.framework.vasu_140m_private_curator_intake import (
    COUNTS,
    FILENAMES,
    _load_jsonl,
    _safe_private_directory,
    validate_private_curator_intake,
)


SCHEMA_ID = "vasu_140m_private_curator_sealing_receipt_v1"
SUITE_ID = "vasu-140m-base-evaluation-v2-curator-candidate-v1"
SCORER_PATH = "evaluation/framework/vasu_140m_base_v2_tasks.py"
RETRIEVED_AT = "2026-08-03T00:00:00+00:00"
Encryptor = Callable[[bytes, str, Path], None]


def _safe_relative(value: str, label: str) -> str:
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"{label} must be a safe repository-relative path")
    return normalized


def _write_new(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _jsonl(records: list[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json(record) + b"\n" for record in records)


def _binding(path: Path, relative: str, record_count: int) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "record_count": record_count,
        "byte_count": path.stat().st_size,
    }


def _age_encrypt(payload: bytes, recipient: str, output: Path) -> None:
    completed = subprocess.run(
        ["age", "-r", recipient, "-o", str(output)],
        input=payload,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Age encryption failed: {message}")


def _recipient_fingerprint(recipient: str) -> str:
    if not recipient.startswith("age1") or len(recipient) < 20:
        raise ValueError("recipient must be an Age X25519 public recipient")
    return hashlib.sha256(recipient.encode("utf-8")).hexdigest()


def _operation(prompt: str) -> str:
    for symbol, name in (
        (" + ", "addition"),
        (" - ", "subtraction"),
        (" * ", "multiplication"),
        (" × ", "multiplication"),
        (" / ", "division"),
        (" ÷ ", "division"),
    ):
        if symbol in prompt:
            return name
    raise ValueError("arithmetic operation is not supported")


def _authoring(record: Mapping[str, object]) -> dict[str, object]:
    dimension = str(record["dimension"])
    family = str(record["semantic_family_id"])
    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("curator provenance must be an object")
    common: dict[str, object] = {
        "schema_id": AUTHORING_SCHEMA_ID,
        "item_id": record["item_id"],
        "dimension": dimension,
        "semantic_family_id": family,
        "parent_document_id": record["parent_document_id"],
        "provenance": {
            "source_name": provenance["source_name"],
            "source_url": provenance["source_url"],
            "license_name": provenance["license_name"],
            "license_url": provenance["license_url"],
            "source_revision": provenance["source_revision"],
            "citation": (
                f"{provenance['citation']} Authored by: "
                f"{provenance['authored_by']}."
            ),
            "retrieved_at": RETRIEVED_AT,
            "human_authored": True,
        },
        "human_approved": True,
    }
    if dimension == "factuality":
        common.update(
            strata={
                "semantic_family": family,
                "task_family": "independent_curator_factuality",
            },
            content={"prompt": record["prompt"], "choices": record["choices"]},
            scoring={"correct_choice_id": record["correct_choice_id"]},
        )
    elif dimension == "arithmetic":
        operation = _operation(str(record["prompt"]))
        common.update(
            strata={
                "semantic_family": family,
                "operation": operation,
                "difficulty": "independent_curator",
                "template_family": f"independent_curator_{operation}_v1",
            },
            content={"prompt": record["prompt"]},
            scoring={
                "answer_type": record["answer_type"],
                "expected_answer": record["expected_answer"],
            },
        )
    elif dimension == "repetition":
        common.update(
            strata={
                "semantic_family": family,
                "prompt_family": "independent_curator_continuation",
            },
            content={"prompt": record["prompt"]},
            scoring={"loop_ngram_size": record["loop_ngram_size"]},
        )
    elif dimension == "robustness":
        common.update(
            strata={
                "semantic_family": family,
                "variant_kind": "independent_curator_perturbation",
            },
            content={
                "baseline_prompt": record["baseline_prompt"],
                "variant_prompt": record["variant_prompt"],
            },
            scoring={"accepted_answers": record["accepted_answers"]},
        )
    else:
        common.update(
            strata={
                "semantic_family": family,
                "category": "independent_curator_manual_review",
            },
            content={"prompt": record["prompt"]},
            scoring={"rubric_dimensions": record["rubric_dimensions"]},
        )
    return common


def _records(
    source: list[dict[str, object]], dimension: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    payloads: list[dict[str, object]] = []
    provenances: list[dict[str, object]] = []
    contaminations: list[dict[str, object]] = []
    for record in sorted(source, key=lambda item: str(item["item_id"])):
        payload, provenance, contamination = build_item_records(_authoring(record))
        task = dict(payload["task"])
        task["split"] = "held_out"
        payload = {**payload, "task": task}
        validate_task(task)
        payloads.append(payload)
        provenances.append(provenance)
        contaminations.append(contamination)
    return payloads, provenances, contaminations


def _validate_receipt(
    receipt: Mapping[str, object], observed: Mapping[str, object]
) -> tuple[str, str]:
    if receipt.get("schema_id") != observed["schema_id"]:
        raise ValueError("curator intake receipt schema mismatch")
    for field in (
        "files",
        "total_records",
        "unique_item_ids",
        "unique_semantic_families",
        "unique_parent_documents",
        "unique_prompt_commitments",
        "development_overlap_count",
        "report_sha256",
    ):
        if receipt.get(field) != observed[field]:
            raise ValueError(f"curator intake receipt mismatch: {field}")
    for flag in (
        "plaintext_copied_to_repository",
        "private_key_opened",
        "held_out_opening_authorized",
        "evaluation_run_authorized",
        "training_authorized",
    ):
        if receipt.get(flag) is not False:
            raise ValueError(f"curator intake receipt flag must remain false: {flag}")
    recipient = str(receipt.get("recipient", ""))
    fingerprint = _recipient_fingerprint(recipient)
    if receipt.get("recipient_fingerprint_sha256") != fingerprint:
        raise ValueError("recipient fingerprint mismatch")
    return recipient, fingerprint


def seal_private_curator_inputs(
    *,
    repository_root: Path,
    private_directory: Path,
    intake_receipt_path: str,
    output_directory: str,
    repository_commit: str,
    encryptor: Encryptor = _age_encrypt,
    expected_counts: Mapping[str, int] = COUNTS,
) -> dict[str, object]:
    """Create one immutable encrypted candidate suite without opening a key."""

    root = repository_root.resolve()
    if len(repository_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        raise ValueError("repository_commit must be a lowercase Git commit")
    private = _safe_private_directory(root, private_directory)
    receipt_relative = _safe_relative(intake_receipt_path, "intake_receipt_path")
    receipt_path = (root / receipt_relative).resolve()
    if not receipt_path.is_relative_to(root) or not receipt_path.is_file():
        raise ValueError("intake receipt path is missing or unsafe")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, Mapping):
        raise ValueError("intake receipt must be an object")
    observed = validate_private_curator_intake(
        root, private, expected_counts=expected_counts
    )
    recipient, fingerprint = _validate_receipt(receipt, observed)

    relative_output = _safe_relative(output_directory, "output_directory")
    destination = (root / relative_output).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("output directory escapes repository")
    if destination.exists():
        raise FileExistsError("sealed candidate output already exists")
    if not destination.parent.is_dir():
        raise ValueError("sealed candidate output parent must already exist")
    staging = destination.with_name(f".{destination.name}.tmp")
    if staging.exists():
        raise FileExistsError("stale sealed candidate staging exists")
    scorer = root / SCORER_PATH
    scorer_sha = sha256_file(scorer)
    staging.mkdir()
    manifests: dict[str, dict[str, object]] = {}
    try:
        for dimension in sorted(expected_counts):
            source = _load_jsonl(private / FILENAMES[dimension])
            payloads, provenances, contaminations = _records(source, dimension)
            dimension_dir = staging / dimension
            dimension_dir.mkdir()
            payload_path = dimension_dir / "payload.jsonl.age"
            encryptor(_jsonl(payloads), recipient, payload_path)
            with payload_path.open("rb") as handle:
                if handle.readline() != b"age-encryption.org/v1\n":
                    raise ValueError("encrypted payload does not have an Age v1 header")
            provenance_path = dimension_dir / "provenance.jsonl"
            contamination_path = dimension_dir / "contamination.jsonl"
            _write_new(provenance_path, _jsonl(provenances))
            _write_new(contamination_path, _jsonl(contaminations))
            final_base = f"{relative_output}/{dimension}"
            count = len(payloads)
            manifest: dict[str, object] = {
                "schema_id": INVENTORY_SCHEMA_ID,
                "inventory_id": f"{SUITE_ID}-{dimension}-held-out",
                "suite_id": SUITE_ID,
                "repository_commit": repository_commit,
                "dimension": dimension,
                "split": "held_out",
                "interface": INTERFACES[dimension],
                "tokenizer_sha256": TOKENIZER_SHA256,
                "payload": _binding(
                    payload_path, f"{final_base}/payload.jsonl.age", count
                ),
                "provenance_index": _binding(
                    provenance_path, f"{final_base}/provenance.jsonl", count
                ),
                "contamination_index": _binding(
                    contamination_path, f"{final_base}/contamination.jsonl", count
                ),
                "scorer": {"path": SCORER_PATH, "sha256": scorer_sha},
                "generation_profile_ids": (
                    []
                    if RESULT_MODES[dimension] in ({"direct_likelihood"}, {"manual"})
                    else ["greedy-v1", "sampled-v1"]
                ),
                "item_commitments": [
                    {
                        "item_id": str(payload["task"]["item_id"]),
                        "task_sha256": task_identity(payload["task"]),
                        "provenance_sha256": provenance_identity(provenance),
                    }
                    for payload, provenance in zip(
                        payloads, provenances, strict=True
                    )
                ],
                "access": {
                    "state": "sealed",
                    "payload_format": "encrypted_jsonl",
                    "encryption_algorithm": "age-x25519",
                    "recipient_fingerprint": fingerprint,
                    "opening_authorized": False,
                },
                "fixture_only": False,
                "production_suite_frozen": False,
                "evaluation_run_authorized": False,
                "training_authorized": False,
            }
            manifest["inventory_sha256"] = inventory_identity(manifest)
            validate_inventory_manifest(manifest)
            _write_new(
                dimension_dir / "manifest.json",
                json.dumps(
                    manifest, indent=2, sort_keys=True, ensure_ascii=False
                ).encode("utf-8")
                + b"\n",
            )
            manifests[dimension] = manifest
        result: dict[str, object] = {
            "schema_id": SCHEMA_ID,
            "suite_id": SUITE_ID,
            "repository_commit": repository_commit,
            "intake_receipt_path": receipt_relative,
            "intake_receipt_sha256": sha256_file(receipt_path),
            "intake_report_sha256": observed["report_sha256"],
            "recipient_fingerprint_sha256": fingerprint,
            "inventory_sha256s": {
                dimension: manifest["inventory_sha256"]
                for dimension, manifest in sorted(manifests.items())
            },
            "record_counts": dict(expected_counts),
            "plaintext_payload_persisted": False,
            "private_key_opened": False,
            "production_suite_frozen": False,
            "evaluation_run_authorized": False,
            "training_authorized": False,
        }
        result["receipt_sha256"] = hashlib.sha256(canonical_json(result)).hexdigest()
        _write_new(
            staging / "sealing_receipt.json",
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        os.replace(staging, destination)
        for manifest in manifests.values():
            validate_inventory_manifest_files(manifest, root)
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
