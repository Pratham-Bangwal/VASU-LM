import torch

from vasu.model import RMSNorm


def main():

    x = torch.randn(4, 16, 384)

    norm = RMSNorm(384)

    y = norm(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    assert x.shape == y.shape

    print("✅ RMSNorm works!")


if __name__ == "__main__":
    main()