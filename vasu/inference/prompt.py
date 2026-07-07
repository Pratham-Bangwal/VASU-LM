def build_prompt(
    user_message: str,
    system_prompt: str | None = None,
) -> str:

    prompt = ""

    if system_prompt:
        prompt += (
            f"System: {system_prompt}\n\n"
        )

    prompt += (
        f"User: {user_message}\n"
        "Assistant:"
    )

    return prompt