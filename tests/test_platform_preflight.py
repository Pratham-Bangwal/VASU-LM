import json
from pathlib import Path

import pytest

from vasu.training.platform_preflight import preflight_configuration


def test_preflight_hash_binds_non_authorizing_config(tmp_path: Path) -> None:
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "training_authorized": False,
                "parent_checkpoint": {"path": str(parent), "sha256": ""},
                "technical_gates": {"cuda_smoke": "not_completed"},
            }
        ),
        encoding="utf-8",
    )
    report = preflight_configuration(config)
    assert report["readiness"] == "review_only_not_launchable"
    assert report["identities"]["parent_checkpoint"]["sha256"]


def test_preflight_rejects_authorizing_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training_authorized": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="only"):
        preflight_configuration(config)
