from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from vasu.data.preparation.wikimedia import (
    load_preparation_config,
    validate_preparation_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/preparation/wikimedia_vasu_140m_likelihood_source_v1.json"


def test_likelihood_source_profile_accepts_separate_bounded_limits() -> None:
    config = load_preparation_config(CONFIG)
    assert config.preparation_profile == "vasu_140m_likelihood_source_v1"
    assert config.max_accepted_parent_documents == 3_000
    assert config.max_output_tokens == 15_000_000


def test_pilot_profile_does_not_inherit_likelihood_limits() -> None:
    config = load_preparation_config(CONFIG)
    with pytest.raises(ValueError, match="hard pilot limit"):
        validate_preparation_config(replace(config, preparation_profile="pilot_v1"))


def test_unknown_preparation_profile_fails_closed() -> None:
    config = load_preparation_config(CONFIG)
    with pytest.raises(ValueError, match="unsupported"):
        validate_preparation_config(replace(config, preparation_profile="unknown"))
