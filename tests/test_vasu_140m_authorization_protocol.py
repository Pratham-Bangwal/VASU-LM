from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from vasu.data.vasu_140m_authorization_protocol import (
    ASSIGNMENT_SHA256,
    AUTHORIZATION_SCOPE,
    AUTHORIZATION_V2_SCHEMA_ID,
    IMPLEMENTATION_ANCHOR_COMMIT,
    IMPLEMENTATION_PATH,
    IMPLEMENTATION_SHA256,
    PROTOCOL_PATH,
    RELEASE_ID,
    authorization_lock_path,
    exclusive_authorization_lock,
    load_detached_authorization,
    publish_with_detached_authorization,
    validate_authorization_v2,
)
from vasu.data.vasu_140m_production_release import (
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_RELEASE_PATH,
    RECEIPT_DIRECTORY,
    PreparedRelease,
    validate_published_release,
)
from vasu.data.vasu_140m_records import sha256_json

from test_vasu_140m_production_release import COMMIT, prepared_release


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def protocol_repository(tmp_path: Path) -> tuple[Path, str]:
    source_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repository"
    implementation = repository / IMPLEMENTATION_PATH
    protocol = repository / PROTOCOL_PATH
    implementation.parent.mkdir(parents=True)
    shutil.copyfile(source_root / IMPLEMENTATION_PATH, implementation)
    shutil.copyfile(source_root / PROTOCOL_PATH, protocol)
    protocol_sha256 = hashlib.sha256(protocol.read_bytes()).hexdigest()
    return repository, protocol_sha256


def v2_prepared_release() -> PreparedRelease:
    prepared = prepared_release()
    qualification = dict(prepared.qualification)
    qualification["implementation_sha256"] = IMPLEMENTATION_SHA256
    qualification["assignment_sha256"] = ASSIGNMENT_SHA256
    body = dict(qualification)
    body.pop("qualification_sha256")
    qualification["qualification_sha256"] = sha256_json(body)
    return PreparedRelease(
        artifacts=prepared.artifacts,
        internal_manifest=prepared.internal_manifest,
        external_manifest=prepared.external_manifest,
        qualification=qualification,
    )


def authorization_v2(
    repository: Path,
    protocol_sha256: str,
) -> dict[str, object]:
    prepared = v2_prepared_release()
    authorization_id = "test-v2-build-001"
    value = {
        "schema_id": AUTHORIZATION_V2_SCHEMA_ID,
        "authorization_id": authorization_id,
        "release_id": RELEASE_ID,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "approved_by": "human test approver",
        "approval_date": "2026-07-30",
        "expires_date": "2026-08-06",
        "implementation_anchor_commit": IMPLEMENTATION_ANCHOR_COMMIT,
        "runtime_commit": COMMIT,
        "implementation_path": IMPLEMENTATION_PATH,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "authorization_protocol_path": PROTOCOL_PATH,
        "authorization_protocol_sha256": protocol_sha256,
        "qualification_sha256": prepared.qualification["qualification_sha256"],
        "assignment_sha256": ASSIGNMENT_SHA256,
        "external_manifest_template_sha256": prepared.qualification[
            "external_manifest_template_sha256"
        ],
        "receipt_path": f"{RECEIPT_DIRECTORY}/{authorization_id}.json",
        "publication_path": PRODUCTION_RELEASE_PATH,
        "external_manifest_path": PRODUCTION_MANIFEST_PATH,
        "publication_overwrite_allowed": False,
        "training_authorized": False,
    }
    value["authorization_sha256"] = sha256_json(value)
    return value


def clean_probe(_root: Path) -> tuple[str, bool]:
    return COMMIT, True


def valid_ancestry(_root: Path, _ancestor: str, _descendant: str) -> bool:
    return True


def test_detached_loader_requires_external_canonical_regular_file(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    external = tmp_path / "authorization.json"
    external.write_bytes(canonical_bytes(value))
    assert load_detached_authorization(external, repository) == value

    internal = repository / "authorization.json"
    internal.write_bytes(canonical_bytes(value))
    with pytest.raises(ValueError, match="outside"):
        load_detached_authorization(internal, repository)

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        load_detached_authorization(noncanonical, repository)


def test_v2_authorization_binds_schema_identities_and_runtime(tmp_path: Path) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    prepared = v2_prepared_release()
    valid = authorization_v2(repository, protocol_sha256)
    receipt = validate_authorization_v2(
        authorization=valid,
        prepared=prepared,
        repository_root=repository,
        protocol_sha256=protocol_sha256,
        current_date=date(2026, 7, 30),
        repository_probe=clean_probe,
        ancestry_probe=valid_ancestry,
    )
    assert receipt == repository / valid["receipt_path"]

    for field in (
        "schema_id",
        "release_id",
        "authorization_scope",
        "implementation_anchor_commit",
        "runtime_commit",
        "implementation_sha256",
        "authorization_protocol_sha256",
        "qualification_sha256",
        "assignment_sha256",
        "external_manifest_template_sha256",
        "receipt_path",
        "publication_path",
        "external_manifest_path",
        "publication_overwrite_allowed",
        "training_authorized",
    ):
        changed = dict(valid)
        changed[field] = not changed[field] if isinstance(changed[field], bool) else "x"
        with pytest.raises(ValueError):
            validate_authorization_v2(
                authorization=changed,
                prepared=prepared,
                repository_root=repository,
                protocol_sha256=protocol_sha256,
                current_date=date(2026, 7, 30),
                repository_probe=clean_probe,
                ancestry_probe=valid_ancestry,
            )

    extra = dict(valid)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        validate_authorization_v2(
            authorization=extra,
            prepared=prepared,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
        )


@pytest.mark.parametrize(
    ("approval", "expires", "today", "message"),
    [
        ("2026-07-31", "2026-08-01", date(2026, 7, 30), "future"),
        ("2026-07-20", "2026-07-29", date(2026, 7, 30), "expired"),
        ("2026-07-30", "2026-08-07", date(2026, 7, 30), "too long"),
    ],
)
def test_v2_authorization_rejects_invalid_date_windows(
    tmp_path: Path,
    approval: str,
    expires: str,
    today: date,
    message: str,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    value["approval_date"] = approval
    value["expires_date"] = expires
    body = dict(value)
    body.pop("authorization_sha256")
    value["authorization_sha256"] = sha256_json(body)
    with pytest.raises(ValueError, match=message):
        validate_authorization_v2(
            authorization=value,
            prepared=v2_prepared_release(),
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            current_date=today,
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
        )


def test_v2_authorization_requires_clean_head_and_anchor(tmp_path: Path) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    prepared = v2_prepared_release()
    cases = (
        (lambda _root: ("b" * 40, True), valid_ancestry, "runtime commit"),
        (lambda _root: (COMMIT, False), valid_ancestry, "clean"),
        (clean_probe, lambda _root, _a, _d: False, "ancestor"),
    )
    for repository_probe, ancestry_probe, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_authorization_v2(
                authorization=value,
                prepared=prepared,
                repository_root=repository,
                protocol_sha256=protocol_sha256,
                repository_probe=repository_probe,
                ancestry_probe=ancestry_probe,
            )


def test_v2_authorization_rejects_code_mutation_and_consumed_receipt(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    prepared = v2_prepared_release()
    (repository / PROTOCOL_PATH).write_text("# mutation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol file identity"):
        validate_authorization_v2(
            authorization=value,
            prepared=prepared,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
        )
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / PROTOCOL_PATH,
        repository / PROTOCOL_PATH,
    )
    receipt = repository / str(value["receipt_path"])
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="consumed"):
        validate_authorization_v2(
            authorization=value,
            prepared=prepared,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
        )


def test_v2_rejects_unsafe_id_and_unreviewed_prepared_identity(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    value["authorization_id"] = "../escape"
    with pytest.raises(ValueError, match="unsafe syntax"):
        validate_authorization_v2(
            authorization=value,
            prepared=v2_prepared_release(),
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
        )

    value = authorization_v2(repository, protocol_sha256)
    with pytest.raises(ValueError, match="prepared qualification implementation"):
        validate_authorization_v2(
            authorization=value,
            prepared=prepared_release(),
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
        )

def test_exclusive_lock_rejects_concurrency_and_recovers_proven_stale_lock(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))
    lock = authorization_lock_path(envelope)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)

    with exclusive_authorization_lock(
        envelope_path=envelope,
        authorization=value,
        now=now,
    ):
        assert lock.is_file()
        with pytest.raises(FileExistsError, match="already exists"):
            with exclusive_authorization_lock(
                envelope_path=envelope,
                authorization=value,
                now=now,
            ):
                pass
    assert not lock.exists()

    stale_body = {
        "schema_id": "vasu.production-release-authorization-lock.v1",
        "authorization_id": value["authorization_id"],
        "authorization_sha256": value["authorization_sha256"],
        "created_at": (now - timedelta(hours=25)).isoformat(),
        "pid": 999999,
    }
    stale_body["lock_sha256"] = sha256_json(stale_body)
    lock.write_bytes(canonical_bytes(stale_body))
    with exclusive_authorization_lock(
        envelope_path=envelope,
        authorization=value,
        now=now,
        allow_stale_recovery=True,
        process_is_alive=lambda _pid: False,
    ):
        assert lock.is_file()
    assert not lock.exists()


def test_stale_lock_recovery_rejects_live_or_mismatched_authority(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))
    lock = authorization_lock_path(envelope)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    stale_body = {
        "schema_id": "vasu.production-release-authorization-lock.v1",
        "authorization_id": value["authorization_id"],
        "authorization_sha256": value["authorization_sha256"],
        "created_at": (now - timedelta(hours=25)).isoformat(),
        "pid": 123,
    }
    stale_body["lock_sha256"] = sha256_json(stale_body)
    lock.write_bytes(canonical_bytes(stale_body))
    with pytest.raises(ValueError, match="still active"):
        with exclusive_authorization_lock(
            envelope_path=envelope,
            authorization=value,
            now=now,
            allow_stale_recovery=True,
            process_is_alive=lambda _pid: True,
        ):
            pass

    stale_body["authorization_sha256"] = "f" * 64
    stale_body.pop("lock_sha256")
    stale_body["lock_sha256"] = sha256_json(stale_body)
    lock.write_bytes(canonical_bytes(stale_body))
    with pytest.raises(ValueError, match="another authority"):
        with exclusive_authorization_lock(
            envelope_path=envelope,
            authorization=value,
            now=now,
            allow_stale_recovery=True,
            process_is_alive=lambda _pid: False,
        ):
            pass


def test_v2_transaction_persists_direct_authorization_binding(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))
    receipt = publish_with_detached_authorization(
        prepared=v2_prepared_release(),
        envelope_path=envelope,
        repository_root=repository,
        protocol_sha256=protocol_sha256,
        current_date=date(2026, 7, 30),
        repository_probe=clean_probe,
        ancestry_probe=valid_ancestry,
        free_bytes=lambda _root: 10**9,
    )
    assert receipt["authorization_sha256"] == value["authorization_sha256"]
    assert receipt["authorization_protocol_sha256"] == protocol_sha256
    persisted = json.loads(
        (repository / str(value["receipt_path"])).read_text(encoding="utf-8")
    )
    assert persisted == receipt
    assert not authorization_lock_path(envelope).exists()
    validate_published_release(repository)


def test_v2_transaction_failure_releases_lock_and_cleans_staging(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))

    def fail_replace(_source: Path, _target: Path) -> None:
        raise PermissionError("injected open-handle failure")

    with pytest.raises(PermissionError, match="open-handle"):
        publish_with_detached_authorization(
            prepared=v2_prepared_release(),
            envelope_path=envelope,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            current_date=date(2026, 7, 30),
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
            free_bytes=lambda _root: 10**9,
            replace_directory=fail_replace,
        )
    assert not authorization_lock_path(envelope).exists()
    assert not (repository / PRODUCTION_RELEASE_PATH).exists()
    assert not list(
        (repository / "data/processed/vasu_140m").glob(
            ".instruction_seed_v1.staging-*"
        )
    )


def test_v2_transaction_rejects_disk_exhaustion_before_writes(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))
    with pytest.raises(OSError, match="insufficient disk"):
        publish_with_detached_authorization(
            prepared=v2_prepared_release(),
            envelope_path=envelope,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            current_date=date(2026, 7, 30),
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
            free_bytes=lambda _root: 0,
        )
    assert not (repository / PRODUCTION_RELEASE_PATH).exists()
    assert not authorization_lock_path(envelope).exists()


def test_v2_transaction_detects_mutation_before_publication(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))

    def mutate(staging: Path) -> None:
        with (staging / "train.tokens.bin").open("ab") as handle:
            handle.write(b"\x00\x00")

    with pytest.raises(ValueError, match="mutation detected"):
        publish_with_detached_authorization(
            prepared=v2_prepared_release(),
            envelope_path=envelope,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            current_date=date(2026, 7, 30),
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
            free_bytes=lambda _root: 10**9,
            mutation_hook=mutate,
        )
    assert not (repository / PRODUCTION_RELEASE_PATH).exists()
    assert not authorization_lock_path(envelope).exists()


def test_v2_postrename_failure_is_quarantined_without_receipt(
    tmp_path: Path,
) -> None:
    repository, protocol_sha256 = protocol_repository(tmp_path)
    value = authorization_v2(repository, protocol_sha256)
    envelope = tmp_path / "authorization.json"
    envelope.write_bytes(canonical_bytes(value))

    def fail_manifest(_source: Path, _target: Path) -> None:
        raise PermissionError("injected manifest failure")

    with pytest.raises(PermissionError, match="manifest failure"):
        publish_with_detached_authorization(
            prepared=v2_prepared_release(),
            envelope_path=envelope,
            repository_root=repository,
            protocol_sha256=protocol_sha256,
            current_date=date(2026, 7, 30),
            repository_probe=clean_probe,
            ancestry_probe=valid_ancestry,
            free_bytes=lambda _root: 10**9,
            replace_manifest=fail_manifest,
        )
    assert (repository / PRODUCTION_RELEASE_PATH).is_dir()
    assert not (repository / PRODUCTION_MANIFEST_PATH).exists()
    assert not (repository / str(value["receipt_path"])).exists()
    assert not authorization_lock_path(envelope).exists()
