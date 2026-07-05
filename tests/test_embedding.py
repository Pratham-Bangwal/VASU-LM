import torch

from vasu.model import TokenEmbedding


def main():

    vocab_size = 32000
    embedding_dim = 384

    embedding = TokenEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(2, 8),
    )

    output = embedding(input_ids)

    print("=" * 50)
    print("Input Shape :", input_ids.shape)
    print("Output Shape:", output.shape)
    print("=" * 50)

    assert output.shape == (2, 8, embedding_dim)

    print("✅ TokenEmbedding is working!")


if __name__ == "__main__":
    main()