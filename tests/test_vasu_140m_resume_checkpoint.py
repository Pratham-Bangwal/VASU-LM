from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from vasu.training.vasu_140m_real_data_resume import STATE_COMPONENTS, state_sha256
from vasu.training.vasu_140m_resume_checkpoint import (
    CHECKPOINT_SCHEMA_ID,
    build_checkpoint_payload,
    load_verified_qualification_checkpoint,
    validate_checkpoint_payload,
    validate_sidecar,
    write_qualification_checkpoint,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
COMMIT = "c" * 40


def _state() -> dict[str, object]:
    return {
        name: {"name": name, "tensor": torch.tensor([1.0, 2.0])}
        for name in STATE_COMPONENTS
    }


def _payload(phase: str = "partial_accumulation") -> dict[str, object]:
    state = _state()
    if phase == "source_boundary":
        state["gradients"] = {}
    return build_checkpoint_payload(
        qualification_id="qualification-fixture-v1",
        specification_sha256=SHA_A,
        repository_commit=COMMIT,
        creation_phase=phase,
        identities={"release": SHA_B, "schedule": SHA_A},
        progress={
            "consumed_record_ids": ["record-0001"],
            "source_index": 0,
            "record_index": 1,
            "microbatch_count": 1,
            "optimizer_update_count": 0,
            "accumulated_microbatches": 1 if phase == "partial_accumulation" else 0,
            "supervised_target_count": 512,
        },
        state=state,
    )


def _root() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="vasu_140m_resume_checkpoint_test_")


def test_payload_is_strict_and_state_bound() -> None:
    payload = _payload()
    assert payload["schema_id"] == CHECKPOINT_SCHEMA_ID
    assert payload["state_sha256"] == state_sha256(payload["state"])
    validate_checkpoint_payload(payload)

    changed = copy.deepcopy(payload)
    changed["state"]["model"]["tensor"][0] = 99
    with pytest.raises(ValueError, match="state identity"):
        validate_checkpoint_payload(changed)


@pytest.mark.parametrize("phase", ["partial_accumulation", "source_boundary"])
def test_round_trip_is_exact_and_non_overwriting(phase: str) -> None:
    with _root() as raw_root:
        root = Path(raw_root)
        payload = _payload(phase)
        sidecar = write_qualification_checkpoint(root, "resume.pt", payload)
        restored, observed = load_verified_qualification_checkpoint(
            root,
            "resume.pt",
            expected_specification_sha256=SHA_A,
            expected_identities={"release": SHA_B, "schedule": SHA_A},
        )
        assert state_sha256(restored) == state_sha256(payload)
        assert observed == sidecar
        validate_sidecar(sidecar)
        with pytest.raises(FileExistsError):
            write_qualification_checkpoint(root, "resume.pt", payload)


def test_rejects_non_system_temporary_root(tmp_path: Path) -> None:
    fake = Path.cwd() / "not-system-temporary"
    fake.mkdir(exist_ok=True)
    try:
        with pytest.raises(ValueError, match="system temporary"):
            write_qualification_checkpoint(fake, "resume.pt", _payload())
    finally:
        fake.rmdir()


@pytest.mark.parametrize("filename", ["../resume.pt", "nested/resume.pt", "resume.bin"])
def test_rejects_unsafe_filename(filename: str) -> None:
    with _root() as raw_root:
        with pytest.raises(ValueError, match="local .pt filename"):
            write_qualification_checkpoint(Path(raw_root), filename, _payload())


def test_rejects_insufficient_disk_without_writing() -> None:
    with _root() as raw_root:
        root = Path(raw_root)
        with pytest.raises(OSError, match="insufficient disk"):
            write_qualification_checkpoint(
                root,
                "resume.pt",
                _payload(),
                minimum_free_bytes=2**63,
            )
        assert list(root.iterdir()) == []


def test_first_atomic_replace_failure_leaves_no_artifact() -> None:
    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected rename failure")

    with _root() as raw_root:
        root = Path(raw_root)
        with pytest.raises(OSError, match="injected rename"):
            write_qualification_checkpoint(root, "resume.pt", _payload(), replace=fail_replace)
        assert list(root.iterdir()) == []


def test_second_atomic_replace_failure_preserves_checkpoint_evidence() -> None:
    calls = 0

    def fail_second(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected sidecar rename failure")
        os.replace(source, destination)

    with _root() as raw_root:
        root = Path(raw_root)
        with pytest.raises(OSError, match="sidecar rename"):
            write_qualification_checkpoint(root, "resume.pt", _payload(), replace=fail_second)
        assert (root / "resume.pt").is_file()
        assert not (root / "resume.pt.sha256.json").exists()
        with pytest.raises(FileNotFoundError, match="both exist"):
            load_verified_qualification_checkpoint(
                root,
                "resume.pt",
                expected_specification_sha256=SHA_A,
                expected_identities={"release": SHA_B, "schedule": SHA_A},
            )


def test_rejects_corrupted_checkpoint_before_returning_state() -> None:
    with _root() as raw_root:
        root = Path(raw_root)
        write_qualification_checkpoint(root, "resume.pt", _payload())
        with (root / "resume.pt").open("ab") as handle:
            handle.write(b"corruption")
        with pytest.raises(ValueError, match="byte identity"):
            load_verified_qualification_checkpoint(
                root,
                "resume.pt",
                expected_specification_sha256=SHA_A,
                expected_identities={"release": SHA_B, "schedule": SHA_A},
            )


def test_rejects_noncanonical_or_corrupt_sidecar() -> None:
    with _root() as raw_root:
        root = Path(raw_root)
        write_qualification_checkpoint(root, "resume.pt", _payload())
        sidecar_path = root / "resume.pt.sha256.json"
        value = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        with pytest.raises(ValueError, match="not canonical"):
            load_verified_qualification_checkpoint(
                root,
                "resume.pt",
                expected_specification_sha256=SHA_A,
                expected_identities={"release": SHA_B, "schedule": SHA_A},
            )


def test_rejects_specification_or_bound_identity_mismatch() -> None:
    with _root() as raw_root:
        root = Path(raw_root)
        write_qualification_checkpoint(root, "resume.pt", _payload())
        with pytest.raises(ValueError, match="specification identity"):
            load_verified_qualification_checkpoint(
                root,
                "resume.pt",
                expected_specification_sha256=SHA_B,
                expected_identities={"release": SHA_B, "schedule": SHA_A},
            )
        with pytest.raises(ValueError, match="bound identities"):
            load_verified_qualification_checkpoint(
                root,
                "resume.pt",
                expected_specification_sha256=SHA_A,
                expected_identities={"release": SHA_A, "schedule": SHA_A},
            )


def test_rejects_missing_partial_gradients_contract() -> None:
    payload = _payload()
    payload["state"]["gradients"] = {}
    payload["state_sha256"] = state_sha256(payload["state"])
    with pytest.raises(ValueError, match="missing gradients"):
        validate_checkpoint_payload(payload)


def test_rejects_linked_checkpoint_when_supported() -> None:
    with _root() as raw_root, _root() as external_root:
        root = Path(raw_root)
        external = Path(external_root) / "external.pt"
        external.write_bytes(b"external")
        link = root / "resume.pt"
        try:
            link.symlink_to(external)
        except OSError:
            pytest.skip("symlink creation is unavailable")
        (root / "resume.pt.sha256.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="link or junction"):
            load_verified_qualification_checkpoint(
                root,
                "resume.pt",
                expected_specification_sha256=SHA_A,
                expected_identities={"release": SHA_B, "schedule": SHA_A},
            )
