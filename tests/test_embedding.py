import torch

from vasu.config import ModelConfig
from vasu.model import TokenEmbedding


config = ModelConfig()


def main():

    embedding = TokenEmbedding(
        config,
    )

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    output = embedding(input_ids)

    print("=" * 50)
    print("Input Shape :", input_ids.shape)
    print("Output Shape:", output.shape)
    print("=" * 50)

    assert output.shape == (2, 8, config.dim)

    print("✅ TokenEmbedding is working!")


if __name__ == "__main__":
    main()