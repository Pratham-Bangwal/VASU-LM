import torch

from vasu.model import MultiHeadAttention


def main():

    attention = MultiHeadAttention(
        dim=384,
        num_heads=6,
        dropout=0.1,
    )

    x = torch.randn(
        2,
        256,
        384,
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