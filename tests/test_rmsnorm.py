import torch

from config.config import ModelConfig
from vasu.model import RMSNorm


config = ModelConfig()


def main():

    x = torch.randn(4, 16, config.dim)

    norm = RMSNorm(config.dim)

    y = norm(x)

    print("Input :", x.shape)
    print("Output:", y.shape)

    assert x.shape == y.shape

    print("✅ RMSNorm works!")


if __name__ == "__main__":
    main()