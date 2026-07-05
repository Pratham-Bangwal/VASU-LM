import torch

from vasu.config import ModelConfig
from vasu.model import MultiHeadAttention


config = ModelConfig()


def main():

    attention = MultiHeadAttention(
        config,
    )

    x = torch.randn(
        2,
        config.max_seq_len,
        config.dim,
    )

    y = attention(x)

    print("=" * 50)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print("=" * 50)

    assert y.shape == x.shape

    print("✅ MultiHeadAttention works!")


if __name__ == "__main__":
    main()