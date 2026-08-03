from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.framework.vasu_140m_base_v2_inventory import (
    validate_inventory_manifest_files,
)
from evaluation.framework.vasu_140m_private_curator_intake import (
    APPROVED_STATUS,
    COUNTS,
    FILENAMES,
    RECORD_SCHEMA_ID,
    validate_private_curator_intake,
)
from evaluation.framework.vasu_140m_private_curator_sealing import (
    SCORER_PATH,
    seal_private_curator_inputs,
)


COMMIT = "1" * 40
RECIPIENT = "age1qualificationrecipient000000000000000000000000000000000"
SMALL_COUNTS = {dimension: 1 for dimension in COUNTS}
ROOT = Path(__file__).resolve().parents[1]


def _provenance() -> dict[str, str]:
    return {
        "source_name": "Independent curator fixture",
        "source_url": "https://example.org/source",
        "license_name": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "source_revision": "v1",
        "citation": "Independent qualification material.",
        "authored_by": "curator-fixture",
    }


def _record(dimension: str) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_id": RECORD_SCHEMA_ID,
        "status": APPROVED_STATUS,
        "item_id": f"sealed-{dimension}-001",
        "dimension": dimension,
        "semantic_family_id": f"sealed-family-{dimension}-001",
        "parent_document_id": f"sealed-parent-{dimension}-001",
        "provenance": _provenance(),
    }
    if dimension == "factuality":
        value.update(
            prompt="Which synthetic option is designated correct?",
            choices=[
                {"choice_id": "a", "text": "Alpha"},
                {"choice_id": "b", "text": "Beta"},
            ],
            correct_choice_id="a",
        )
    elif dimension == "arithmetic":
        value.update(
            prompt="Calculate exactly: 29 * 13 = ?",
            answer_type="integer",
            expected_answer="377",
        )
    elif dimension == "repetition":
        value.update(
            prompt="Beyond the quiet workshop a synthetic narrative continued",
            loop_ngram_size=3,
        )
    elif dimension == "robustness":
        value.update(
            baseline_prompt="Synthetic baseline expects nine.",
            variant_prompt="synthetic VARIANT expects exactly nine!",
            accepted_answers=["9", "nine"],
        )
    else:
        value.update(
            prompt="Explain a synthetic process for manual review.",
            rubric_dimensions=["coherence", "factual_support", "degeneration"],
        )
    return value


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "repository"
    private = tmp_path / "private"
    private.mkdir()
    (repository / "evaluation/candidates").mkdir(parents=True)
    scorer = repository / SCORER_PATH
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_text("# scorer fixture\n", encoding="utf-8", newline="\n")
    development_suite = (
        repository
        / "evaluation/fixtures/vasu_140m_assistant_authored_internal_v1"
    )
    for dimension in COUNTS:
        development = development_suite / dimension
        development.mkdir(parents=True)
        content = (
            {
                "baseline_prompt": f"Development baseline {dimension}",
                "variant_prompt": f"Development variant {dimension}",
            }
            if dimension == "robustness"
            else {"prompt": f"Development prompt {dimension}"}
        )
        _write(development / "payload.jsonl", {"content": content})
        _write(private / FILENAMES[dimension], _record(dimension))
    intake = validate_private_curator_intake(
        repository, private, expected_counts=SMALL_COUNTS
    )
    receipt = {
        **intake,
        "recipient": RECIPIENT,
        "recipient_fingerprint_sha256": hashlib.sha256(
            RECIPIENT.encode("utf-8")
        ).hexdigest(),
    }
    receipt_path = repository / "configs/evaluation/intake.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return repository, private, receipt_path


def _encrypt(payload: bytes, recipient: str, output: Path) -> None:
    assert recipient == RECIPIENT
    output.write_bytes(
        b"age-encryption.org/v1\nfixture:" + hashlib.sha256(payload).digest()
    )


def _seal(repository: Path, private: Path):
    return seal_private_curator_inputs(
        repository_root=repository,
        private_directory=private,
        intake_receipt_path="configs/evaluation/intake.json",
        output_directory="evaluation/candidates/sealed-v1",
        repository_commit=COMMIT,
        encryptor=_encrypt,
        expected_counts=SMALL_COUNTS,
    )


def test_seals_five_opaque_non_authorizing_inventories(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)
    report = _seal(repository, private)
    assert report["record_counts"] == SMALL_COUNTS
    assert report["plaintext_payload_persisted"] is False
    assert report["private_key_opened"] is False
    assert report["production_suite_frozen"] is False
    assert report["training_authorized"] is False
    output = repository / "evaluation/candidates/sealed-v1"
    for dimension in COUNTS:
        manifest = json.loads(
            (output / dimension / "manifest.json").read_text(encoding="utf-8")
        )
        validate_inventory_manifest_files(manifest, repository)
        assert manifest["split"] == "held_out"
        assert manifest["fixture_only"] is False
        assert manifest["production_suite_frozen"] is False
        ciphertext = (output / dimension / "payload.jsonl.age").read_bytes()
        assert ciphertext.startswith(b"age-encryption.org/v1\n")
        source = _record(dimension)
        plaintext_values = (
            [source["baseline_prompt"], source["variant_prompt"]]
            if dimension == "robustness"
            else [source["prompt"]]
        )
        assert all(str(value).encode() not in ciphertext for value in plaintext_values)
        assert not (output / dimension / "payload.jsonl").exists()


def test_frozen_qualification_binds_current_implementation_and_tests() -> None:
    report = json.loads(
        (
            ROOT
            / "evaluation/fixtures/"
            "vasu_140m_private_curator_sealing_qualification_v1.json"
        ).read_text(encoding="utf-8")
    )

    def lf_sha(path: Path) -> str:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    assert report["implementation_sha256"] == lf_sha(
        ROOT / "evaluation/framework/vasu_140m_private_curator_sealing.py"
    )
    assert report["tests_sha256"] == lf_sha(
        ROOT / "tests/test_vasu_140m_private_curator_sealing.py"
    )
    assert report["record_count"] == 1500
    assert report["sealing_invoked"] is False
    assert report["private_key_opened"] is False
    assert report["training_authorized"] is False


def test_refuses_overwrite(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)
    _seal(repository, private)
    with pytest.raises(FileExistsError, match="already exists"):
        _seal(repository, private)


def test_rejects_mutated_private_source_against_receipt(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)
    record = copy.deepcopy(_record("repetition"))
    record["prompt"] = "A changed but structurally valid private prompt"
    _write(private / FILENAMES["repetition"], record)
    with pytest.raises(ValueError, match="receipt mismatch: files"):
        _seal(repository, private)


def test_rejects_recipient_fingerprint_substitution(tmp_path: Path) -> None:
    repository, private, receipt_path = _workspace(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["recipient_fingerprint_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="recipient fingerprint"):
        _seal(repository, private)


def test_encryption_failure_removes_staging_and_preserves_no_output(
    tmp_path: Path,
) -> None:
    repository, private, _ = _workspace(tmp_path)

    def fail(payload: bytes, recipient: str, output: Path) -> None:
        output.write_bytes(b"partial")
        raise RuntimeError("injected encryption failure")

    with pytest.raises(RuntimeError, match="injected encryption failure"):
        seal_private_curator_inputs(
            repository_root=repository,
            private_directory=private,
            intake_receipt_path="configs/evaluation/intake.json",
            output_directory="evaluation/candidates/sealed-v1",
            repository_commit=COMMIT,
            encryptor=fail,
            expected_counts=SMALL_COUNTS,
        )
    assert not (repository / "evaluation/candidates/sealed-v1").exists()
    assert not (repository / "evaluation/candidates/.sealed-v1.tmp").exists()


def test_rejects_non_age_encryptor_output(tmp_path: Path) -> None:
    repository, private, _ = _workspace(tmp_path)

    def invalid(payload: bytes, recipient: str, output: Path) -> None:
        output.write_bytes(b"not-age")

    with pytest.raises(ValueError, match="Age v1 header"):
        seal_private_curator_inputs(
            repository_root=repository,
            private_directory=private,
            intake_receipt_path="configs/evaluation/intake.json",
            output_directory="evaluation/candidates/sealed-v1",
            repository_commit=COMMIT,
            encryptor=invalid,
            expected_counts=SMALL_COUNTS,
        )
