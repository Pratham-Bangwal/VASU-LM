import pytest

from evaluation.comparison import paired_binary_bootstrap


def test_paired_bootstrap_is_reproducible_and_reports_delta() -> None:
    result = paired_binary_bootstrap(
        [False, False, True, True], [True, False, True, True], samples=1000, seed=7
    )
    assert result["delta"] == 0.25
    assert result == paired_binary_bootstrap(
        [False, False, True, True], [True, False, True, True], samples=1000, seed=7
    )


def test_paired_bootstrap_rejects_unpaired_input() -> None:
    with pytest.raises(ValueError, match="paired"):
        paired_binary_bootstrap([True], [])
