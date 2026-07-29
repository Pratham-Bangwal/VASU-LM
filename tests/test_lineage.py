import json
from pathlib import Path

from vasu.utils.lineage import build_lineage_index


def test_lineage_index_is_read_only_and_hash_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "data/manifests/example.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"training_authorized": False}), encoding="utf-8")
    result = build_lineage_index(tmp_path)
    assert result["read_only"] is True
    assert result["artifact_count"] == 1
    assert result["artifacts"][0]["training_authorized"] is False
    assert result["artifacts"][0]["sha256"]
