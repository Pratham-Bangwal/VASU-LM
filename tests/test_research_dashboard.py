from pathlib import Path

from scripts.report_vasu_research_dashboard import build_dashboard


def test_dashboard_is_read_only_and_includes_candidate_e() -> None:
    dashboard = build_dashboard(Path("."))
    assert dashboard["read_only"] is True
    assert dashboard["lineage"]["artifact_count"] > 0
    assert dashboard["candidate_e"]["training_authorized"] is False
