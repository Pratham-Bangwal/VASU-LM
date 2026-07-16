"""Interactive chat entry point for the preferred VASU-60M assistant."""

import time
from pathlib import Path

import torch

from vasu.config import get_vasu_60m_config
from vasu.inference.generate import generate
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer


DEFAULT_CHECKPOINT = Path(
    "checkpoints/vasu_60m/alpaca_masked_v3_from_200k/best.pt"
)
TOKENIZER_PATH = Path("assets/tokenizer.json")
PROMPT_FORMAT = "alpaca"

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.45
TOP_K = 20
TOP_P = 0.8
USE_KV_CACHE = False


def clean_response(response: str) -> str:
    """Remove accidental prompt-template continuations."""

    stop_markers = (
        "User:",
        "Assistant:",
        "### Instruction:",
        "### Response:",
        "Instruction:",
        "Response:",
    )

    for marker in stop_markers:
        if marker in response:
            response = response.split(marker, maxsplit=1)[0]

    return response.strip()


def load_model(
    device: torch.device,
) -> tuple[VASUModel, VASUTokenizer]:
    """Load the preferred VASU-60M assistant checkpoint."""

    if not TOKENIZER_PATH.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    if not DEFAULT_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {DEFAULT_CHECKPOINT}"
        )

    tokenizer = VASUTokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    config = get_vasu_60m_config()
    model = VASUModel(config).to(device)

    checkpoint = torch.load(
        DEFAULT_CHECKPOINT,
        map_location=device,
    )

    if "model" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain the required 'model' state."
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()

    return model, tokenizer


def main() -> None:
    """Run an interactive terminal chat session."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, tokenizer = load_model(device)

    print("VASU-60M ready. Type something, or type 'exit' to stop.")
    print(f"Device: {device}")
    print(f"Checkpoint: {DEFAULT_CHECKPOINT}")

    while True:
        prompt = input("\nYou: ").strip()

        if prompt.lower() == "exit":
            break

        if not prompt:
            continue

        start_time = time.time()

        with torch.inference_mode():
            response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                prompt_format=PROMPT_FORMAT,
                use_kv_cache=USE_KV_CACHE,
            )

        response = clean_response(response)

        elapsed = time.time() - start_time

        print(f"\nGeneration time: {elapsed:.2f}s")
        print("\nVASU:", response)


if __name__ == "__main__":
    main()
