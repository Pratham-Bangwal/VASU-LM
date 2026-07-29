"""Print Candidate E's non-authorizing governance preflight report."""

from __future__ import annotations

import json
from pathlib import Path

from vasu.training.experiment_governance import sha256_file


EVIDENCE = (
    "docs/CANDIDATE_E_INDEPENDENT_REVIEW_PACKET.md",
    "docs/CANDIDATE_E_BUDGET_AND_EVALUATION_PROTOCOL.md",
    "docs/CANDIDATE_E_PRERELEASE_ACCEPTANCE.md",
    "docs/CANDIDATE_E_REFERENCE_TOKENIZER_AUDIT.md",
)


def build_report(repository_root: Path) -> dict[str, object]:
    """Return the current review state without generating a Candidate E release."""

    evidence = {}
    for relative in EVIDENCE:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    release_root = repository_root / "data/processed/capability"
    candidate_e_release_exists = (
        any(
            child.is_dir() and "candidate_e" in child.name
            for child in release_root.glob("*")
        )
        if release_root.is_dir()
        else False
    )
    return {
        "format_version": "vasu_candidate_e_governance_preflight_v1",
        "experiment_id": "candidate_e_step_supervision_matched_pair",
        "training_authorized": False,
        "readiness": "review_evidence_complete_release_not_approved",
        "evidence": evidence,
        "candidate_e_release_exists": candidate_e_release_exists,
        "next_required_action": "independent_review_of_new_logical_data_specification",
    }


def main() -> None:
    print(json.dumps(build_report(Path.cwd()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
