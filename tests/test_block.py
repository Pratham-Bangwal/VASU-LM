import torch

from config.config import ModelConfig
from vasu.model import TransformerBlock


config = ModelConfig()


def main():

    block = TransformerBlock(
        dim=config.dim,
        num_heads=config.n_heads,
        hidden_dim=config.hidden_dim,
    )

    x = torch.randn(
        2,
        config.max_seq_len,
        config.dim,
    )

    y = block(x)

    print("=" * 50)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print("=" * 50)

    assert y.shape == x.shape

    print("✅ Transformer Block works!")


if __name__ == "__main__":
    main()