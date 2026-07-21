import torch

from vasu.config import ModelConfig
from vasu.model import VASUModel


def test_vasu_model_logits_shape():
    config = ModelConfig(
        vocab_size=100,
        max_seq_len=8,
        dim=24,
        n_heads=4,
        n_layers=2,
        hidden_dim=48,
    )

    model = VASUModel(config)
    x = torch.randint(
        0,
        config.vocab_size,
        (2, config.max_seq_len),
    )

    logits = model(x)

    assert logits.shape == (
        2,
        config.max_seq_len,
        config.vocab_size,
    )
