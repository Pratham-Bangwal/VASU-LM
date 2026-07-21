"""Prompt templates shared by dataset preparation and inference."""

from collections.abc import Callable


PLAIN_PROMPT_FORMAT = "plain"
ALPACA_PROMPT_FORMAT = "alpaca"
SUPPORTED_PROMPT_FORMATS = (
    PLAIN_PROMPT_FORMAT,
    ALPACA_PROMPT_FORMAT,
)


def format_alpaca_prompt(
    instruction: str,
    input_text: str | None = None,
    include_response_header: bool = True,
) -> str:
    """Match the exact User/Assistant template in the existing alpaca.bin."""
    prompt = f"User: {instruction}"

    if input_text is not None and input_text.strip():
        prompt += f"\n{input_text}"

    if include_response_header:
        prompt += "\nAssistant:"

    return prompt


def format_alpaca_example(
    instruction: str,
    response: str,
    input_text: str | None = None,
) -> str:
    """Serialize one example exactly as the existing preparation script did."""
    prompt = format_alpaca_prompt(
        instruction=instruction,
        input_text=input_text,
        include_response_header=True,
    )
    return f"{prompt} {response}\n"


def format_plain_prompt(
    instruction: str,
    input_text: str | None = None,
) -> str:
    """Return an unwrapped continuation prompt."""
    if input_text is not None and input_text.strip():
        return f"{instruction}\n{input_text}"
    return instruction


def get_prompt_formatter(
    prompt_format: str,
) -> Callable[[str, str | None], str]:
    """Resolve a declared prompt format or reject it clearly."""
    if prompt_format == ALPACA_PROMPT_FORMAT:
        return lambda instruction, input_text=None: format_alpaca_prompt(
            instruction,
            input_text,
            include_response_header=True,
        )
    if prompt_format == PLAIN_PROMPT_FORMAT:
        return format_plain_prompt

    valid = ", ".join(SUPPORTED_PROMPT_FORMATS)
    raise ValueError(
        f"Unknown prompt format '{prompt_format}'. Valid formats: {valid}."
    )


def format_prompt(
    instruction: str,
    prompt_format: str,
    input_text: str | None = None,
) -> str:
    """Format one prompt using a named, validated template."""
    formatter = get_prompt_formatter(prompt_format)
    return formatter(instruction, input_text)
