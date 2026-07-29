import json
from pathlib import Path

import pytest

from vasu.training.experiment_governance import build_review_packet


def _json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_packet_is_hash_bound_and_non_authorizing(tmp_path: Path) -> None:
    release = _json(tmp_path / "release.json", {"training_authorized": False})
    evaluation = _json(tmp_path / "evaluation.json", {"score": 0.0})
    packet = build_review_packet(
        experiment_id="candidate_e",
        release_manifests={"control": release},
        schedule_manifest=None,
        configuration=None,
        evaluation_snapshots={"arithmetic": evaluation},
    )
    assert packet["training_authorized"] is False
    assert packet["release_manifests"]["control"]["sha256"]


def test_packet_rejects_authorizing_input(tmp_path: Path) -> None:
    release = _json(tmp_path / "release.json", {"training_authorized": True})
    evaluation = _json(tmp_path / "evaluation.json", {})
    with pytest.raises(ValueError, match="must explicitly"):
        build_review_packet(
            experiment_id="candidate_e",
            release_manifests={"control": release},
            schedule_manifest=None,
            configuration=None,
            evaluation_snapshots={"arithmetic": evaluation},
        )
