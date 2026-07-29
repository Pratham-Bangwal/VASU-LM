from pathlib import Path

from scripts.report_candidate_e_governance_preflight import build_report


def test_candidate_e_preflight_is_non_authorizing_and_evidence_bound() -> None:
    report = build_report(Path("."))
    assert report["training_authorized"] is False
    assert report["readiness"] == "review_evidence_complete_release_not_approved"
    assert report["evidence"]
