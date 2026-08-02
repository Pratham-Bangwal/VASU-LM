from __future__ import annotations

import copy
from datetime import date
import json
import os
from pathlib import Path

import pytest

from vasu.data.vasu_140m_production_release import (
    AUTHORIZATION_SCHEMA_ID,
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_RELEASE_PATH,
    RECEIPT_DIRECTORY,
    PreparedRelease,
    authorized_external_manifest,
    publication_status,
    publish_authorized_release,
    qualify_compiled_release,
    qualify_production_release,
    validate_authorization,
    validate_prepared_release,
    validate_published_release,
    validate_qualification_report,
)
from vasu.data.vasu_140m_records import compile_text_example, sha256_json


COMMIT = "a" * 40
REVIEW_BASE_COMMIT = "20d79c3f1be58596c22b53c9ce87df7943b8a90c"
DECISIONS = {
    "plan_decision_sha256": "1" * 64,
    "fixture_decision_sha256": "2" * 64,
    "design_decision_sha256": "3" * 64,
}


def clean_repository_probe(_root: Path) -> tuple[str, bool]:
    return COMMIT, True


class CharacterTokenizer:
    def encode(self, text: str) -> list[int]:
        return [10 + ord(character) % 200 for character in text]


def prepared_release() -> PreparedRelease:
    tokenizer = CharacterTokenizer()
    splits = {}
    lineage = {}
    for split in ("train", "development", "evaluation"):
        examples = []
        for index in range(2):
            example_id = f"{split}-{index}"
            examples.append(
                compile_text_example(
                    tokenizer=tokenizer,
                    example_id=example_id,
                    split=split,
                    prompt=f"Prompt {split} {index}:",
                    response=f" answer {index}",
                )
            )
            lineage[example_id] = {
                "source_id": f"source-{index % 2}",
                "capability": "test",
            }
        splits[split] = examples
    return qualify_compiled_release(
        splits=splits,
        lineage=lineage,
        decision_identities=DECISIONS,
        repository_commit=COMMIT,
        implementation_sha256="4" * 64,
        enforce_production_counts=False,
    )


def authorization(prepared: PreparedRelease) -> dict[str, object]:
    authorization_id = "test-build-001"
    value = {
        "schema_id": AUTHORIZATION_SCHEMA_ID,
        "authorization_id": authorization_id,
        "scope": "one_production_release_build",
        "approved_by": "test approver",
        "approval_date": "2026-07-30",
        # The generic authorization fixture is intentionally long-lived so
        # unrelated behavior tests do not depend on the wall clock. Expiry is
        # verified separately below with an explicit current_date.
        "expires_date": "9999-12-31",
        "repository_commit": COMMIT,
        "implementation_sha256": prepared.qualification["implementation_sha256"],
        "plan_id": "vasu_140m_instruction_seed_v1",
        "plan_sha256": prepared.qualification["plan_sha256"],
        "qualification_sha256": prepared.qualification["qualification_sha256"],
        "expected_artifact_evidence": prepared.qualification["artifact_evidence"],
        "external_manifest_sha256": authorized_external_manifest(
            prepared, authorization_id
        )["manifest_sha256"],
        "release_directory": PRODUCTION_RELEASE_PATH,
        "external_manifest_path": PRODUCTION_MANIFEST_PATH,
        "receipt_path": f"{RECEIPT_DIRECTORY}/{authorization_id}.json",
        "overwrite_allowed": False,
        "training_authorized": False,
    }
    value["authorization_sha256"] = sha256_json(value)
    return value


def test_qualification_is_deterministic_and_read_only(tmp_path: Path) -> None:
    first = prepared_release()
    second = prepared_release()
    assert first == second
    assert first.qualification["production_release_created"] is False
    assert first.qualification["release_build_permitted"] is False
    assert first.qualification["training_authorized"] is False
    assert not (tmp_path / PRODUCTION_RELEASE_PATH).exists()
    assert not (tmp_path / PRODUCTION_MANIFEST_PATH).exists()
    validate_prepared_release(first)


def test_real_source_qualification_matches_frozen_review_identity() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    if (repository_root / PRODUCTION_RELEASE_PATH).exists():
        qualification = json.loads(
            (
                repository_root
                / "evaluation/fixtures/"
                "vasu_140m_instruction_seed_v1_production_qualification.json"
            ).read_text(encoding="utf-8")
        )
    else:
        qualification = qualify_production_release(
            repository_root=repository_root,
            repository_commit=REVIEW_BASE_COMMIT,
        ).qualification
    assert qualification["qualification_sha256"] == (
        "ec44ee9a8a50125f516c330b54ee8605ba124f87cd0aee13737c8718ccadb24e"
    )
    assert qualification["implementation_sha256"] == (
        "0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2"
    )
    assert qualification["assignment_sha256"] == (
        "59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394"
    )
    assert qualification["split_counts"] == {
        "train": 898,
        "development": 48,
        "evaluation": 50,
    }
    assert qualification["source_audit"]["decoded_round_trip_count"] == 996
    validate_qualification_report(qualification)


def test_complete_mask_audit_rejects_serialized_prompt_supervision() -> None:
    prepared = prepared_release()
    artifacts = dict(prepared.artifacts)
    mask = bytearray(artifacts["train.mask.bin"])
    mask[0] = 1
    artifacts["train.mask.bin"] = bytes(mask)
    altered = PreparedRelease(
        artifacts=artifacts,
        internal_manifest=prepared.internal_manifest,
        external_manifest=prepared.external_manifest,
        qualification=prepared.qualification,
    )
    with pytest.raises(ValueError, match="does not bind artifact"):
        validate_prepared_release(altered)


def test_authorization_binds_every_release_identity(tmp_path: Path) -> None:
    prepared = prepared_release()
    valid = authorization(prepared)
    receipt = validate_authorization(
        authorization=valid,
        prepared=prepared,
        repository_commit=COMMIT,
        repository_root=tmp_path,
    )
    assert receipt == tmp_path / valid["receipt_path"]
    for field in (
        "repository_commit",
        "implementation_sha256",
        "plan_sha256",
        "qualification_sha256",
        "expected_artifact_evidence",
        "external_manifest_sha256",
        "release_directory",
        "external_manifest_path",
        "overwrite_allowed",
        "training_authorized",
    ):
        changed = copy.deepcopy(valid)
        changed[field] = "wrong" if not isinstance(changed[field], bool) else True
        with pytest.raises(ValueError, match=field):
            validate_authorization(
                authorization=changed,
                prepared=prepared,
                repository_commit=COMMIT,
                repository_root=tmp_path,
            )
    expired = authorization(prepared)
    expired["expires_date"] = "2026-07-29"
    body = dict(expired)
    body.pop("authorization_sha256")
    expired["authorization_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match="expired"):
        validate_authorization(
            authorization=expired,
            prepared=prepared,
            repository_commit=COMMIT,
            repository_root=tmp_path,
            current_date=date(2026, 7, 30),
        )
    mutated = authorization(prepared)
    mutated["approved_by"] = "different approver"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_authorization(
            authorization=mutated,
            prepared=prepared,
            repository_commit=COMMIT,
            repository_root=tmp_path,
        )


def test_publication_requires_observed_clean_exact_commit(tmp_path: Path) -> None:
    prepared = prepared_release()
    for probe, message in (
        (lambda _root: ("b" * 40, True), "commit"),
        (lambda _root: (COMMIT, False), "clean"),
    ):
        with pytest.raises(ValueError, match=message):
            publish_authorized_release(
                prepared=prepared,
                authorization=authorization(prepared),
                repository_root=tmp_path,
                repository_commit=COMMIT,
                repository_probe=probe,
                free_bytes=lambda _: 10**9,
            )


def test_transactional_publication_and_authorization_reuse_rejection(
    tmp_path: Path,
) -> None:
    prepared = prepared_release()
    approved = authorization(prepared)
    receipt = publish_authorized_release(
        prepared=prepared,
        authorization=approved,
        repository_root=tmp_path,
        repository_commit=COMMIT,
        repository_probe=clean_repository_probe,
        free_bytes=lambda _: 10**9,
    )
    release = tmp_path / PRODUCTION_RELEASE_PATH
    external = tmp_path / PRODUCTION_MANIFEST_PATH
    assert receipt["release_complete"] is True
    assert release.is_dir()
    assert set(path.name for path in release.iterdir()) == set(prepared.artifacts)
    assert json.loads(external.read_text()) == authorized_external_manifest(
        prepared, str(approved["authorization_id"])
    )
    assert (tmp_path / approved["receipt_path"]).is_file()
    assert publication_status(tmp_path) == "complete"
    validate_published_release(tmp_path)
    with pytest.raises(ValueError, match="already been consumed"):
        validate_authorization(
            authorization=approved,
            prepared=prepared,
            repository_commit=COMMIT,
            repository_root=tmp_path,
        )


def test_published_release_tampering_is_detected(tmp_path: Path) -> None:
    prepared = prepared_release()
    publish_authorized_release(
        prepared=prepared,
        authorization=authorization(prepared),
        repository_root=tmp_path,
        repository_commit=COMMIT,
        repository_probe=clean_repository_probe,
        free_bytes=lambda _: 10**9,
    )
    with (tmp_path / PRODUCTION_RELEASE_PATH / "evaluation.mask.bin").open(
        "ab"
    ) as handle:
        handle.write(b"\x00")
    with pytest.raises(ValueError, match="mask identity mismatch"):
        validate_published_release(tmp_path)


def test_insufficient_disk_space_publishes_nothing(tmp_path: Path) -> None:
    prepared = prepared_release()
    with pytest.raises(OSError, match="insufficient disk"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 0,
        )
    assert not (tmp_path / PRODUCTION_RELEASE_PATH).exists()
    assert not (tmp_path / PRODUCTION_MANIFEST_PATH).exists()


def test_prepublication_open_handle_failure_cleans_staging(tmp_path: Path) -> None:
    prepared = prepared_release()

    def locked(_source: Path, _target: Path) -> None:
        raise PermissionError("simulated open handle")

    with pytest.raises(PermissionError, match="open handle"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
            replace_directory=locked,
        )
    assert not (tmp_path / PRODUCTION_RELEASE_PATH).exists()
    assert not list((tmp_path / "data/processed/vasu_140m").glob(".*.staging-*"))


def test_post_validation_mutation_is_detected(tmp_path: Path) -> None:
    prepared = prepared_release()

    def mutate(staging: Path) -> None:
        with (staging / "train.tokens.bin").open("ab") as handle:
            handle.write(b"\x00\x00")

    with pytest.raises(ValueError, match="mutation detected"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
            mutation_hook=mutate,
        )
    assert not (tmp_path / PRODUCTION_RELEASE_PATH).exists()


def test_post_rename_mutation_is_quarantined_before_manifest(tmp_path: Path) -> None:
    prepared = prepared_release()

    def replace_then_mutate(source: Path, target: Path) -> None:
        os.replace(source, target)
        with (target / "train.mask.bin").open("ab") as handle:
            handle.write(b"\x00")

    with pytest.raises(ValueError, match="mutation detected"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
            replace_directory=replace_then_mutate,
        )
    assert publication_status(tmp_path) == "quarantined_incomplete"
    assert not (tmp_path / PRODUCTION_MANIFEST_PATH).exists()


def test_external_manifest_failure_leaves_quarantined_directory(
    tmp_path: Path,
) -> None:
    prepared = prepared_release()

    def fail_manifest(_source: Path, _target: Path) -> None:
        raise PermissionError("simulated external manifest failure")

    with pytest.raises(PermissionError, match="external manifest"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
            replace_manifest=fail_manifest,
        )
    assert (tmp_path / PRODUCTION_RELEASE_PATH).is_dir()
    assert not (tmp_path / PRODUCTION_MANIFEST_PATH).exists()
    assert not (tmp_path / authorization(prepared)["receipt_path"]).exists()
    assert publication_status(tmp_path) == "quarantined_incomplete"
    assert not list((tmp_path / "data/manifests/vasu_140m").glob("*.tmp"))


def test_existing_output_and_unbound_artifact_fail_closed(tmp_path: Path) -> None:
    prepared = prepared_release()
    existing = tmp_path / PRODUCTION_RELEASE_PATH
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
        )
    altered = PreparedRelease(
        artifacts={**prepared.artifacts, "unexpected.bin": b"x"},
        internal_manifest=prepared.internal_manifest,
        external_manifest=prepared.external_manifest,
        qualification=prepared.qualification,
    )
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    with pytest.raises(ValueError, match="unrecognized release artifact"):
        publish_authorized_release(
            prepared=altered,
            authorization=authorization(altered),
            repository_root=clean_root,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
        )


def test_unresolved_sibling_staging_fails_closed(tmp_path: Path) -> None:
    prepared = prepared_release()
    sibling = (
        tmp_path
        / "data/processed/vasu_140m/.instruction_seed_v1.staging-recovery"
    )
    sibling.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="unresolved sibling"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction defense")
def test_windows_junction_in_protected_path_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepared_release()
    from vasu.data import vasu_140m_production_release as module

    (tmp_path / "data/processed").mkdir(parents=True)
    monkeypatch.setattr(
        module,
        "_is_link_or_junction",
        lambda path: path.name == "processed",
    )
    with pytest.raises(ValueError, match="link or junction"):
        publish_authorized_release(
            prepared=prepared,
            authorization=authorization(prepared),
            repository_root=tmp_path,
            repository_commit=COMMIT,
            repository_probe=clean_repository_probe,
            free_bytes=lambda _: 10**9,
        )
