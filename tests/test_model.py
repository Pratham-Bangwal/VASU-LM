import torch

from vasu.config import ModelConfig
from vasu.model import VASUModel


def main():

    config = ModelConfig()

    model = VASUModel(config)

    x = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )

    logits = model(x)

    print("=" * 60)

    print("Input :", x.shape)

    print("Output:", logits.shape)

    print()

    total = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Parameters: {total:,}")

    print("=" * 60)

    assert logits.shape == (
        2,
        config.max_seq_len,
        config.vocab_size,
    )

    print("✅ VASUModel works!")


if __name__ == "__main__":
    main()