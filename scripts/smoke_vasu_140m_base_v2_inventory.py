"""Qualify VASU-140M evaluation inventory contracts in temporary storage."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.framework.vasu_140m_base_v2 import (  # noqa: E402
    DIMENSIONS,
    INTERFACES,
    RESULT_MODES,
    TOKENIZER_SHA256,
    canonical_json,
    sha256_file,
)
from evaluation.framework.vasu_140m_base_v2_inventory import (  # noqa: E402
    CONTAMINATION_RECORD_SCHEMA_ID,
    INVENTORY_SCHEMA_ID,
    PAYLOAD_RECORD_SCHEMA_ID,
    PROVENANCE_RECORD_SCHEMA_ID,
    content_text_sha256,
    inventory_identity,
    normalized_text_sha256,
    normalized_word_count,
    provenance_identity,
    validate_inventory_manifest_files,
)
from evaluation.framework.vasu_140m_base_v2_tasks import (  # noqa: E402
    TASK_SCHEMA_ID,
    task_identity,
)


QUALIFICATION_SCHEMA_ID = "vasu_140m_base_evaluation_inventory_qualification_v1"
MODULE_PATH = ROOT / "evaluation/framework/vasu_140m_base_v2_inventory.py"
TASK_MODULE_PATH = ROOT / "evaluation/framework/vasu_140m_base_v2_tasks.py"
TEST_PATH = ROOT / "tests/test_vasu_140m_base_v2_inventory.py"


def _line(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"


def _write(root: Path, relative: str, payload: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def _task_and_content(dimension: str) -> tuple[dict[str, object], dict[str, object]]:
    item_id = f"{dimension}-development-fixture-001"
    if dimension == "likelihood":
        context = "Synthetic likelihood fixture "
        target = "text with no benchmark content."
        text = context + target
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": dimension,
            "split": "development",
            "strata": {"source": "synthetic-fixture"},
            "input": {
                "text_sha256": content_text_sha256(text),
                "target_token_count": 8,
            },
            "scoring": {"kind": "token_log_likelihood"},
        }
        content = {
            "context": context,
            "target": target,
            "text_sha256": content_text_sha256(text),
            "target_sha256": content_text_sha256(target),
        }
    elif dimension == "factuality":
        prompt = "Which fixture option is marked correct?"
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": dimension,
            "split": "development",
            "strata": {"task_family": "multiple_choice"},
            "input": {
                "prompt_sha256": content_text_sha256(prompt),
                "choice_ids": ["a", "b"],
            },
            "scoring": {"correct_choice_id": "b"},
        }
        content = {
            "prompt": prompt,
            "prompt_sha256": content_text_sha256(prompt),
            "choices": [
                {"choice_id": "a", "text": "First synthetic option"},
                {"choice_id": "b", "text": "Second synthetic option"},
            ],
        }
    elif dimension == "arithmetic":
        prompt = "What is 20 plus 22?"
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": dimension,
            "split": "development",
            "strata": {
                "operation": "addition",
                "difficulty": "easy",
                "template_family": "direct",
            },
            "input": {"prompt_sha256": content_text_sha256(prompt)},
            "scoring": {"answer_type": "integer", "expected_answer": "42"},
        }
        content = {"prompt": prompt, "prompt_sha256": content_text_sha256(prompt)}
    elif dimension == "repetition":
        prompt = "Continue this synthetic sequence without looping: alpha beta"
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": dimension,
            "split": "development",
            "strata": {"prompt_family": "synthetic-continuation"},
            "input": {"prompt_sha256": content_text_sha256(prompt)},
            "scoring": {"loop_ngram_size": 3},
        }
        content = {"prompt": prompt, "prompt_sha256": content_text_sha256(prompt)}
    elif dimension == "robustness":
        baseline = "Name the synthetic fixture label."
        variant = "  Name   the synthetic fixture label.  "
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": dimension,
            "split": "development",
            "strata": {"variant_kind": "whitespace"},
            "input": {
                "pair_id": "synthetic-pair-001",
                "baseline_prompt_sha256": content_text_sha256(baseline),
                "variant_prompt_sha256": content_text_sha256(variant),
                "variant_kind": "whitespace",
            },
            "scoring": {"accepted_answers": ["fixture"]},
        }
        content = {
            "baseline_prompt": baseline,
            "baseline_prompt_sha256": content_text_sha256(baseline),
            "variant_prompt": variant,
            "variant_prompt_sha256": content_text_sha256(variant),
        }
    else:
        prompt = "Review this synthetic continuation using the frozen rubric."
        task = {
            "schema_id": TASK_SCHEMA_ID,
            "item_id": item_id,
            "dimension": dimension,
            "split": "development",
            "strata": {"category": "synthetic-review"},
            "input": {"prompt_sha256": content_text_sha256(prompt)},
            "scoring": {
                "rubric_dimensions": [
                    "coherence",
                    "factual_support",
                    "degeneration",
                ]
            },
        }
        content = {"prompt": prompt, "prompt_sha256": content_text_sha256(prompt)}
    return task, content


def _provenance(item_id: str) -> dict[str, object]:
    return {
        "schema_id": PROVENANCE_RECORD_SCHEMA_ID,
        "item_id": item_id,
        "source_name": "VASU synthetic inventory fixture",
        "source_url": "https://example.test/vasu-fixture",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "fixture-v1",
        "citation": "Synthetic qualification fixture; no benchmark content.",
        "parent_document_id": f"parent-{item_id}",
        "retrieved_at": "2026-08-01T12:00:00+05:30",
        "human_authored": True,
    }


def _contamination(
    item_id: str,
    task: dict[str, object] | None = None,
    content: dict[str, object] | None = None,
) -> dict[str, object]:
    if task is None or content is None:
        prompts = [f"sealed prompt commitment for {item_id}"]
        answers: list[str] = []
    else:
        dimension = str(task["dimension"])
        scoring = task["scoring"]
        assert isinstance(scoring, dict)
        if dimension == "likelihood":
            prompts = [str(content["context"])]
            answers = [str(content["target"])]
        elif dimension == "factuality":
            prompts = [str(content["prompt"])]
            choices = content["choices"]
            assert isinstance(choices, list)
            answers = [
                str(choice["text"])
                for choice in choices
                if choice["choice_id"] == scoring["correct_choice_id"]
            ]
        elif dimension == "arithmetic":
            prompts = [str(content["prompt"])]
            answers = [str(scoring["expected_answer"])]
        elif dimension == "robustness":
            prompts = [
                str(content["baseline_prompt"]),
                str(content["variant_prompt"]),
            ]
            answers = [str(value) for value in scoring["accepted_answers"]]
        else:
            prompts = [str(content["prompt"])]
            answers = []
    ngram_hashes: set[str] = set()
    for value in [*prompts, *answers]:
        words = value.strip().split()
        for index in range(max(0, len(words) - 8 + 1)):
            ngram_hashes.add(
                normalized_text_sha256(" ".join(words[index : index + 8]))
            )
    prompt_commitments = {
        normalized_text_sha256(value): normalized_word_count(value)
        for value in prompts
    }
    answer_commitments = {
        normalized_text_sha256(value): normalized_word_count(value)
        for value in answers
    }
    return {
        "schema_id": CONTAMINATION_RECORD_SCHEMA_ID,
        "item_id": item_id,
        "parent_document_id": f"parent-{item_id}",
        "prompt_exact_commitments": [
            {"sha256": digest, "word_count": prompt_commitments[digest]}
            for digest in sorted(prompt_commitments)
        ],
        "answer_exact_commitments": [
            {"sha256": digest, "word_count": answer_commitments[digest]}
            for digest in sorted(answer_commitments)
        ],
        "ngram_words": 8,
        "ngram_sha256s": sorted(ngram_hashes),
        "semantic_fingerprint": {
            "method": "synthetic-minhash-v1",
            "value": hashlib.sha256(f"semantic-{item_id}".encode()).hexdigest(),
        },
    }


def _binding(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "record_count": 1,
        "byte_count": path.stat().st_size,
    }


def _manifest(
    root: Path, dimension: str, split: str, repository_commit: str
) -> dict[str, object]:
    item_id = f"{dimension}-{split}-fixture-001"
    base = f"evaluation/inventories/fixture/{split}/{dimension}"
    payload_relative = f"{base}/payload.{'jsonl.age' if split == 'held_out' else 'jsonl'}"
    provenance_relative = f"{base}/provenance.jsonl"
    contamination_relative = f"{base}/contamination.jsonl"
    if split == "development":
        task, content = _task_and_content(dimension)
        item_id = str(task["item_id"])
        payload_body: str | bytes = _line(
            {
                "schema_id": PAYLOAD_RECORD_SCHEMA_ID,
                "task": task,
                "content": content,
            }
        )
        task_sha = task_identity(task)
    else:
        payload_body = (
            b"age-encryption.org/v1\n"
            + f"synthetic-opaque-{dimension}".encode()
        )
        task_sha = hashlib.sha256(f"sealed-task-{dimension}".encode()).hexdigest()
    provenance = _provenance(item_id)
    contamination = _contamination(
        item_id,
        task if split == "development" else None,
        content if split == "development" else None,
    )
    payload_path = _write(root, payload_relative, payload_body)
    provenance_path = _write(root, provenance_relative, _line(provenance))
    contamination_path = _write(root, contamination_relative, _line(contamination))
    scorer_relative = "evaluation/framework/vasu_140m_base_v2_tasks.py"
    scorer_path = root / scorer_relative
    manifest: dict[str, object] = {
        "schema_id": INVENTORY_SCHEMA_ID,
        "inventory_id": f"{dimension}-{split}-fixture-v1",
        "suite_id": "vasu-140m-base-eval-v2-inventory-fixture",
        "repository_commit": repository_commit,
        "dimension": dimension,
        "split": split,
        "interface": INTERFACES[dimension],
        "tokenizer_sha256": TOKENIZER_SHA256,
        "payload": _binding(payload_path, payload_relative),
        "provenance_index": _binding(provenance_path, provenance_relative),
        "contamination_index": _binding(
            contamination_path, contamination_relative
        ),
        "scorer": {"path": scorer_relative, "sha256": sha256_file(scorer_path)},
        "generation_profile_ids": (
            []
            if RESULT_MODES[dimension] in ({"direct_likelihood"}, {"manual"})
            else ["greedy-v1", "sampled-v1"]
        ),
        "item_commitments": [
            {
                "item_id": item_id,
                "task_sha256": task_sha,
                "provenance_sha256": provenance_identity(provenance),
            }
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
                "recipient_fingerprint": "AGE-RECIPIENT-FIXTURE-001",
                "opening_authorized": False,
            }
        ),
        "fixture_only": True,
        "production_suite_frozen": False,
        "evaluation_run_authorized": False,
        "training_authorized": False,
        "inventory_sha256": "0" * 64,
    }
    manifest["inventory_sha256"] = inventory_identity(manifest)
    return manifest


def _qualification_identity(report: dict[str, object]) -> str:
    body = dict(report)
    body.pop("qualification_sha256", None)
    return hashlib.sha256(canonical_json(body)).hexdigest()


def main() -> None:
    repository_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    temporary_root: Path | None = None
    with tempfile.TemporaryDirectory(prefix="vasu_140m_eval_inventory_") as temporary:
        temporary_root = Path(temporary)
        scorer_target = temporary_root / "evaluation/framework/vasu_140m_base_v2_tasks.py"
        scorer_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TASK_MODULE_PATH, scorer_target)
        development: dict[str, str] = {}
        held_out: dict[str, str] = {}
        opening_rejections = 0
        for dimension in sorted(DIMENSIONS):
            dev_manifest = _manifest(
                temporary_root, dimension, "development", repository_commit
            )
            held_out_manifest = _manifest(
                temporary_root, dimension, "held_out", repository_commit
            )
            validate_inventory_manifest_files(dev_manifest, temporary_root)
            validate_inventory_manifest_files(held_out_manifest, temporary_root)
            try:
                validate_inventory_manifest_files(
                    held_out_manifest, temporary_root, open_held_out=True
                )
            except PermissionError:
                opening_rejections += 1
            else:
                raise RuntimeError("held-out opening did not fail closed")
            development[dimension] = str(dev_manifest["inventory_sha256"])
            held_out[dimension] = str(held_out_manifest["inventory_sha256"])
        report: dict[str, object] = {
            "schema_id": QUALIFICATION_SCHEMA_ID,
            "repository_commit": repository_commit,
            "implementation_sha256": sha256_file(MODULE_PATH),
            "tests_sha256": sha256_file(TEST_PATH),
            "smoke_sha256": sha256_file(Path(__file__)),
            "development_inventory_sha256s": development,
            "held_out_inventory_sha256s": held_out,
            "dimension_count": len(DIMENSIONS),
            "development_payloads_validated": len(development),
            "sealed_payloads_validated_without_decryption": len(held_out),
            "held_out_opening_rejections": opening_rejections,
            "fixture_only": True,
            "prompt_content_persisted": False,
            "held_out_opened": False,
            "production_suite_frozen": False,
            "evaluation_run_authorized": False,
            "training_authorized": False,
            "temporary_artifacts_cleaned": False,
            "qualification_sha256": "0" * 64,
        }
    assert temporary_root is not None
    report["temporary_artifacts_cleaned"] = not temporary_root.exists()
    report["qualification_sha256"] = _qualification_identity(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
