import torch

from vasu.config import ModelConfig
from vasu.model import TokenEmbedding


def test_token_embedding_shape():
    config = ModelConfig(vocab_size=100, max_seq_len=8, dim=24)
    embedding = TokenEmbedding(config)

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 8),
    )

    output = embedding(input_ids)

    assert output.shape == (2, 8, config.dim)
