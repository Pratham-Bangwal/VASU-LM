"""Read-only construction preflight for the VASU-140M-v1 family."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from vasu.config import get_vasu_140m_config  # noqa: E402
from vasu.model import (  # noqa: E402
    VASUModel,
    expected_parameter_count,
    get_model_family,
    validate_family_config,
)


def build_preflight_report() -> dict[str, object]:
    """Construct on the meta device and return deterministic gate evidence."""
    config = get_vasu_140m_config()
    family = validate_family_config("vasu_140m_v1", config)
    with torch.device("meta"):
        model = VASUModel(config)

    observed_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    tied_embeddings = (
        model.lm_head.weight is model.embedding.embedding.weight
    )
    checks = {
        "config_identity": family.config_fingerprint
        == get_model_family("vasu_140m_v1").config_fingerprint,
        "parameter_count": observed_parameters
        == family.expected_parameter_count
        == expected_parameter_count(config),
        "head_dimension": config.dim // config.n_heads == 64,
        "weight_tying": tied_embeddings,
        "meta_construction": all(
            parameter.device.type == "meta"
            for parameter in model.parameters()
        ),
    }
    return {
        "schema": "vasu.model-family-preflight.v1",
        "family_id": family.family_id,
        "family_fingerprint": family.family_fingerprint,
        "config_fingerprint": family.config_fingerprint,
        "parameter_count": observed_parameters,
        "max_seq_len": config.max_seq_len,
        "checks": checks,
        "passed": all(checks.values()),
        "training_authorized": False,
        "remaining_gates": [
            "cpu_forward_backward",
            "cuda_memory_throughput_thermal",
            "kv_cache_parity",
            "checkpoint_round_trip",
            "exact_resume",
            "513_token_data_release",
            "frozen_evaluation_baselines",
            "scientific_plan_and_hash_bound_authorization",
        ],
    }


def main() -> int:
    report = build_preflight_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
