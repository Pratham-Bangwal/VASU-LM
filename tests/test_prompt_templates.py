import pytest

from vasu.data.prompt_templates import (
    format_alpaca_example,
    format_alpaca_prompt,
    format_prompt,
)


def test_alpaca_prompt_instruction_only():
    assert format_alpaca_prompt("Explain gravity.") == (
        "User: Explain gravity.\nAssistant:"
    )


def test_alpaca_prompt_instruction_and_input():
    assert format_alpaca_prompt("Summarize this.", "A short passage.") == (
        "User: Summarize this.\nA short passage.\nAssistant:"
    )


def test_alpaca_example_exact_response_boundary_and_newline():
    assert format_alpaca_example("Say hello.", "Hello!") == (
        "User: Say hello.\nAssistant: Hello!\n"
    )


def test_alpaca_prompt_ignores_empty_input():
    assert format_alpaca_prompt("Say hello.", "  \n") == (
        "User: Say hello.\nAssistant:"
    )


def test_unknown_prompt_format_is_rejected():
    with pytest.raises(ValueError, match="Unknown prompt format"):
        format_prompt("hello", prompt_format="unknown")
