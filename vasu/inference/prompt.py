def build_prompt(
    user_message: str,
    system_prompt: str | None = None,
) -> str:

    if system_prompt is None:
        system_prompt = (
            "You are VASU, a helpful AI assistant."
        )

    return (
        f"System: {system_prompt}\n\n"
        f"User: {user_message}\n"
        "Assistant:"
    )