from vasu.data.prompt_templates import format_alpaca_prompt


def build_prompt(
    user_message: str,
    system_prompt: str | None = None,
) -> str:
    """Build the legacy chat prompt using the shared Alpaca formatter."""
    return format_alpaca_prompt(user_message)
