"""Detached v2 authorization gate for one VASU-140M release publication.

This module does not create authorization envelopes. It validates an
operator-supplied envelope outside the repository and invokes the accepted
release builder only after exact identity and exclusive-lock checks pass.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Iterator, Mapping
import uuid

from vasu.data.vasu_140m_production_release import (
    PRODUCTION_MANIFEST_PATH,
    PRODUCTION_RELEASE_PATH,
    RECEIPT_DIRECTORY,
    PreparedRelease,
    _ARTIFACT_NAMES,
    _json_bytes,
    _validate_protected_path,
    _validate_staged_release,
    authorized_external_manifest,
)
from vasu.data.vasu_140m_records import sha256_json


AUTHORIZATION_V2_SCHEMA_ID = "vasu.production-release-authorization.v2"
AUTHORIZATION_SCOPE = "construct_once"
RELEASE_ID = "vasu_140m_instruction_seed_v1"
IMPLEMENTATION_ANCHOR_COMMIT = "c014716ec38ef8f08842356fc016359dc5a233d7"
IMPLEMENTATION_PATH = "vasu/data/vasu_140m_production_release.py"
IMPLEMENTATION_SHA256 = (
    "0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2"
)
ASSIGNMENT_SHA256 = (
    "59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394"
)
MAX_VALIDITY_DAYS = 7
STALE_LOCK_AGE = timedelta(hours=24)
PROTOCOL_PATH = "vasu/data/vasu_140m_authorization_protocol.py"

_HEX_40 = frozenset("0123456789abcdef")
_HEX_64 = frozenset("0123456789abcdef")
_AUTHORIZATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_V2_FIELDS = {
    "schema_id",
    "authorization_id",
    "release_id",
    "authorization_scope",
    "approved_by",
    "approval_date",
    "expires_date",
    "implementation_anchor_commit",
    "runtime_commit",
    "implementation_path",
    "implementation_sha256",
    "authorization_protocol_path",
    "authorization_protocol_sha256",
    "qualification_sha256",
    "assignment_sha256",
    "external_manifest_template_sha256",
    "receipt_path",
    "publication_path",
    "external_manifest_path",
    "publication_overwrite_allowed",
    "training_authorized",
    "authorization_sha256",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hex(value: object, length: int, alphabet: frozenset[str]) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in alphabet for character in value)
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_detached_authorization(
    envelope_path: Path,
    repository_root: Path,
) -> dict[str, object]:
    """Load canonical JSON only when the envelope is outside the repository."""

    root = repository_root.resolve()
    unresolved = envelope_path.absolute()
    if unresolved.is_symlink():
        raise ValueError("authorization envelope must be a regular non-link file")
    path = unresolved.resolve(strict=True)
    if _is_within(path, root):
        raise ValueError("authorization envelope must be outside the repository")
    if not path.is_file() or path.is_symlink():
        raise ValueError("authorization envelope must be a regular non-link file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authorization envelope is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("authorization envelope must contain a JSON object")
    if raw != _canonical_json_bytes(value):
        raise ValueError("authorization envelope is not canonical JSON")
    return value


def probe_repository(repository_root: Path) -> tuple[str, bool]:
    """Return exact HEAD and whether index/worktree are clean."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, not bool(status.strip())


def commit_is_ancestor(
    repository_root: Path,
    ancestor: str,
    descendant: str,
) -> bool:
    """Return whether Git proves the required ancestry relation."""

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repository_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def validate_authorization_v2(
    *,
    authorization: Mapping[str, object],
    prepared: PreparedRelease,
    repository_root: Path,
    protocol_sha256: str,
    current_date: date | None = None,
    repository_probe: Callable[[Path], tuple[str, bool]] = probe_repository,
    ancestry_probe: Callable[[Path, str, str], bool] = commit_is_ancestor,
) -> Path:
    """Validate exact scope, identities, runtime state, and unused receipt."""

    if set(authorization) != _V2_FIELDS:
        raise ValueError("authorization v2 fields are invalid")
    authorization_id = authorization.get("authorization_id")
    approved_by = authorization.get("approved_by")
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("authorization_id is invalid")
    if not _AUTHORIZATION_ID_RE.fullmatch(authorization_id):
        raise ValueError("authorization_id has unsafe syntax")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("approved_by is invalid")
    runtime_commit = authorization.get("runtime_commit")
    if not _is_hex(runtime_commit, 40, _HEX_40):
        raise ValueError("runtime_commit is invalid")
    if not _is_hex(protocol_sha256, 64, _HEX_64):
        raise ValueError("authorization protocol identity is invalid")

    root = repository_root.resolve()
    protocol_path = root / PROTOCOL_PATH
    implementation_path = root / IMPLEMENTATION_PATH
    expected_receipt = (
        f"{RECEIPT_DIRECTORY}/{authorization_id}.json"
    )
    expected = {
        "schema_id": AUTHORIZATION_V2_SCHEMA_ID,
        "release_id": RELEASE_ID,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "implementation_anchor_commit": IMPLEMENTATION_ANCHOR_COMMIT,
        "implementation_path": IMPLEMENTATION_PATH,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "authorization_protocol_path": PROTOCOL_PATH,
        "authorization_protocol_sha256": protocol_sha256,
        "qualification_sha256": prepared.qualification["qualification_sha256"],
        "assignment_sha256": ASSIGNMENT_SHA256,
        "external_manifest_template_sha256": prepared.qualification[
            "external_manifest_template_sha256"
        ],
        "receipt_path": expected_receipt,
        "publication_path": PRODUCTION_RELEASE_PATH,
        "external_manifest_path": PRODUCTION_MANIFEST_PATH,
        "publication_overwrite_allowed": False,
        "training_authorized": False,
    }
    if prepared.qualification.get("implementation_sha256") != IMPLEMENTATION_SHA256:
        raise ValueError("prepared qualification implementation identity mismatch")
    if prepared.qualification.get("assignment_sha256") != ASSIGNMENT_SHA256:
        raise ValueError("prepared qualification assignment identity mismatch")
    for field, expected_value in expected.items():
        if authorization.get(field) != expected_value:
            raise ValueError(f"authorization {field} mismatch")
    if _sha256_file(implementation_path) != IMPLEMENTATION_SHA256:
        raise ValueError("reviewed implementation file identity changed")
    if _sha256_file(protocol_path) != protocol_sha256:
        raise ValueError("authorization protocol file identity changed")

    try:
        approval = date.fromisoformat(str(authorization["approval_date"]))
        expires = date.fromisoformat(str(authorization["expires_date"]))
    except ValueError as error:
        raise ValueError("authorization dates must use YYYY-MM-DD") from error
    today = current_date or date.today()
    if approval > today:
        raise ValueError("authorization approval date is in the future")
    if expires < approval or today > expires:
        raise ValueError("authorization is expired or has an invalid date range")
    if expires - approval > timedelta(days=MAX_VALIDITY_DAYS):
        raise ValueError("authorization validity window is too long")

    body = dict(authorization)
    reported_hash = body.pop("authorization_sha256")
    if not _is_hex(reported_hash, 64, _HEX_64) or sha256_json(body) != reported_hash:
        raise ValueError("authorization identity mismatch")

    observed_commit, clean = repository_probe(root)
    if observed_commit != runtime_commit:
        raise ValueError("runtime commit does not match clean HEAD")
    if not clean:
        raise ValueError("repository worktree and index must be clean")
    if not ancestry_probe(root, IMPLEMENTATION_ANCHOR_COMMIT, str(runtime_commit)):
        raise ValueError("implementation anchor is not an ancestor of runtime commit")

    receipt = root / expected_receipt
    _validate_protected_path(receipt, root)
    if receipt.exists():
        raise ValueError("authorization has already been consumed")
    if (root / PRODUCTION_RELEASE_PATH).exists():
        raise FileExistsError("production release already exists")
    if (root / PRODUCTION_MANIFEST_PATH).exists():
        raise FileExistsError("production manifest already exists")
    return receipt


def authorization_lock_path(envelope_path: Path) -> Path:
    """Return a detached lock path adjacent to the detached envelope."""

    return envelope_path.with_name(f"{envelope_path.name}.publication.lock")


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def exclusive_authorization_lock(
    *,
    envelope_path: Path,
    authorization: Mapping[str, object],
    now: datetime | None = None,
    allow_stale_recovery: bool = False,
    process_is_alive: Callable[[int], bool] = _process_is_alive,
) -> Iterator[Path]:
    """Acquire one detached lock; recover only a proven stale matching lock."""

    lock_path = authorization_lock_path(envelope_path.resolve())
    timestamp = now or datetime.now(timezone.utc)
    lock_body = {
        "schema_id": "vasu.production-release-authorization-lock.v1",
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization["authorization_sha256"],
        "created_at": timestamp.isoformat(),
        "pid": os.getpid(),
    }
    lock_body["lock_sha256"] = sha256_json(lock_body)
    payload = _canonical_json_bytes(lock_body)
    try:
        with lock_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if not allow_stale_recovery:
            raise FileExistsError("authorization publication lock already exists")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        existing_body = dict(existing)
        existing_hash = existing_body.pop("lock_sha256", None)
        if sha256_json(existing_body) != existing_hash:
            raise ValueError("existing authorization lock identity is invalid")
        if (
            existing.get("authorization_id") != authorization["authorization_id"]
            or existing.get("authorization_sha256")
            != authorization["authorization_sha256"]
        ):
            raise ValueError("existing authorization lock belongs to another authority")
        created = datetime.fromisoformat(str(existing["created_at"]))
        if created.tzinfo is None:
            raise ValueError("existing authorization lock timestamp lacks timezone")
        if timestamp - created < STALE_LOCK_AGE:
            raise ValueError("authorization publication lock is not stale")
        pid = int(existing["pid"])
        if process_is_alive(pid):
            raise ValueError("authorization publication lock owner is still active")
        lock_path.unlink()
        with lock_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    try:
        yield lock_path
    finally:
        if lock_path.exists() and lock_path.read_bytes() == payload:
            lock_path.unlink()


def publish_with_detached_authorization(
    *,
    prepared: PreparedRelease,
    envelope_path: Path,
    repository_root: Path,
    protocol_sha256: str,
    current_date: date | None = None,
    repository_probe: Callable[[Path], tuple[str, bool]] = probe_repository,
    ancestry_probe: Callable[[Path, str, str], bool] = commit_is_ancestor,
    allow_stale_lock_recovery: bool = False,
    free_bytes: Callable[[Path], int] | None = None,
    replace_directory: Callable[[Path, Path], None] = os.replace,
    replace_manifest: Callable[[Path, Path], None] = os.replace,
    mutation_hook: Callable[[Path], None] | None = None,
) -> dict[str, object]:
    """Validate v2 authority and publish once through the accepted builder."""

    authorization = load_detached_authorization(envelope_path, repository_root)
    validate_authorization_v2(
        authorization=authorization,
        prepared=prepared,
        repository_root=repository_root,
        protocol_sha256=protocol_sha256,
        current_date=current_date,
        repository_probe=repository_probe,
        ancestry_probe=ancestry_probe,
    )
    with exclusive_authorization_lock(
        envelope_path=envelope_path,
        authorization=authorization,
        allow_stale_recovery=allow_stale_lock_recovery,
    ):
        return _publish_v2_transaction(
            prepared=prepared,
            authorization=authorization,
            repository_root=repository_root.resolve(),
            protocol_sha256=protocol_sha256,
            free_bytes=free_bytes,
            replace_directory=replace_directory,
            replace_manifest=replace_manifest,
            mutation_hook=mutation_hook,
        )


def _publish_v2_transaction(
    *,
    prepared: PreparedRelease,
    authorization: Mapping[str, object],
    repository_root: Path,
    protocol_sha256: str,
    free_bytes: Callable[[Path], int] | None,
    replace_directory: Callable[[Path, Path], None],
    replace_manifest: Callable[[Path, Path], None],
    mutation_hook: Callable[[Path], None] | None,
) -> dict[str, object]:
    """Publish reviewed bytes and persist a receipt bound to v2 authority."""

    root = repository_root
    release = root / PRODUCTION_RELEASE_PATH
    external = root / PRODUCTION_MANIFEST_PATH
    receipt = root / str(authorization["receipt_path"])
    external_temporary = external.with_suffix(".json.tmp")
    receipt_temporary = receipt.with_suffix(".json.tmp")
    for path in (release, external, receipt, external_temporary, receipt_temporary):
        _validate_protected_path(path, root)
        if path.exists():
            raise FileExistsError(f"protected output already exists: {path}")
    staging = release.parent / f".{release.name}.staging-{uuid.uuid4().hex}"
    _validate_protected_path(staging, root)
    if release.parent.exists() and any(
        release.parent.glob(f".{release.name}.staging-*")
    ):
        raise FileExistsError("an unresolved sibling staging directory exists")

    final_manifest = authorized_external_manifest(
        prepared, str(authorization["authorization_id"])
    )
    required_bytes = sum(len(payload) for payload in prepared.artifacts.values())
    required_bytes += len(_json_bytes(final_manifest))
    available = (free_bytes or (lambda path: shutil.disk_usage(path).free))(root)
    if available < required_bytes * 2:
        raise OSError("insufficient disk space for transactional publication")

    release.parent.mkdir(parents=True, exist_ok=True)
    external.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    for path in (release, external, receipt, staging):
        _validate_protected_path(path, root)
    staging.mkdir()
    directory_published = False
    try:
        for name, payload in prepared.artifacts.items():
            if name not in _ARTIFACT_NAMES:
                raise ValueError(f"unrecognized release artifact: {name}")
            artifact = staging / name
            with artifact.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _validate_staged_release(staging, prepared)
        if mutation_hook is not None:
            mutation_hook(staging)
        _validate_staged_release(staging, prepared)
        replace_directory(staging, release)
        directory_published = True
        _validate_staged_release(release, prepared)
        with external_temporary.open("xb") as handle:
            handle.write(_json_bytes(final_manifest))
            handle.flush()
            os.fsync(handle.fileno())
        replace_manifest(external_temporary, external)
        receipt_payload = {
            "schema_id": "vasu.production-release-authorization-receipt.v1",
            "authorization_id": authorization["authorization_id"],
            "qualification_sha256": prepared.qualification["qualification_sha256"],
            "external_manifest_sha256": final_manifest["manifest_sha256"],
            "repository_commit": authorization["runtime_commit"],
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "authorization_sha256": authorization["authorization_sha256"],
            "authorization_protocol_sha256": protocol_sha256,
            "release_complete": True,
            "training_authorized": False,
        }
        receipt_payload["receipt_sha256"] = sha256_json(receipt_payload)
        with receipt_temporary.open("xb") as handle:
            handle.write(_json_bytes(receipt_payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(receipt_temporary, receipt)
        return receipt_payload
    except BaseException:
        if not directory_published and staging.exists():
            shutil.rmtree(staging)
        if external_temporary.exists():
            external_temporary.unlink()
        if receipt_temporary.exists():
            receipt_temporary.unlink()
        raise
