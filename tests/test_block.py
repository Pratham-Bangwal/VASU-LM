import torch

from vasu.config import ModelConfig
from vasu.model import TransformerBlock


def test_transformer_block_preserves_shape():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=8,
        dim=24,
        n_heads=4,
        n_layers=2,
        hidden_dim=48,
    )

    block = TransformerBlock(config)
    x = torch.randn(2, config.max_seq_len, config.dim)

    y = block(x)

    assert y.shape == x.shape
