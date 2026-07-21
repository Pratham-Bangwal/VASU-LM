"""Generate raw continuations from the VASU-60M FineWeb checkpoint."""

from pathlib import Path

import torch

from vasu.config import get_vasu_60m_config
from vasu.model.model import VASUModel
from vasu.tokenizer.tokenizer import VASUTokenizer


CHECKPOINT_PATH = Path(
    "checkpoints/vasu_60m/milestones/fineweb_step_150000.pt"
)
TOKENIZER_PATH = "assets/tokenizer.json"
SEED = 42

MAX_NEW_TOKENS = 60
TEMPERATURE = 0.6
TOP_K = 20
USE_GREEDY = False

SAMPLED_OUTPUT_PATH = Path("evaluation/vasu_60m_base_step_150000.txt")
GREEDY_OUTPUT_PATH = Path(
    "evaluation/vasu_60m_base_step_150000_greedy.txt"
)

PROMPTS = [
    "Artificial intelligence is",
    "Machine learning allows computers to",
    "India is a country",
    "The purpose of education is",
    "Once upon a time",
]


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Sample one token from the final-position logits."""
    logits = logits / max(temperature, 1e-5)

    if top_k > 0:
        top_values, _ = torch.topk(
            logits,
            k=min(top_k, logits.size(-1)),
        )
        cutoff = top_values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.no_grad()
def generate_continuation(
    model: VASUModel,
    tokenizer: VASUTokenizer,
    prompt: str,
    device: torch.device,
) -> str:
    token_ids: list[int] = tokenizer.encode(prompt)

    generated = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    for _ in range(MAX_NEW_TOKENS):
        model_input = generated[:, -model.config.max_seq_len :]
        logits = model(model_input)
        final_position_logits = logits[:, -1, :]
        if USE_GREEDY:
            next_token = torch.argmax(
                final_position_logits,
                dim=-1,
                keepdim=True,
            )
        else:
            next_token = sample_next_token(
                final_position_logits,
                temperature=TEMPERATURE,
                top_k=TOP_K,
            )

        generated = torch.cat((generated, next_token), dim=1)

    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    report_lines: list[str] = []
    generation_mode = "greedy" if USE_GREEDY else "sampled"
    output_path = (
        GREEDY_OUTPUT_PATH if USE_GREEDY else SAMPLED_OUTPUT_PATH
    )

    def record(line: str = "") -> None:
        print(line)
        report_lines.append(line)

    record("VASU-60M Base Evaluation")
    record("=" * 80)
    record(f"Device: {device}")
    record(f"Checkpoint path: {CHECKPOINT_PATH}")
    record(f"Generation mode: {generation_mode}")
    record(
        "Generation settings: "
        f"max_new_tokens={MAX_NEW_TOKENS}, "
        f"temperature={TEMPERATURE}, "
        f"top_k={TOP_K}, "
        f"seed={SEED}"
    )

    tokenizer = VASUTokenizer()
    tokenizer.load(TOKENIZER_PATH)

    config = get_vasu_60m_config()
    model = VASUModel(config).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    global_step = checkpoint.get("global_step", "unknown")
    record(f"Checkpoint global_step: {global_step}")

    for prompt in PROMPTS:
        continuation = generate_continuation(
            model,
            tokenizer,
            prompt,
            device,
        )
        record("")
        record("=" * 80)
        record(f"PROMPT: {prompt}")
        record("-" * 80)
        record(continuation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved evaluation output to: {output_path}")


if __name__ == "__main__":
    main()
