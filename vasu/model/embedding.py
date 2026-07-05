"""
Token embedding layer for VASU.

Converts token IDs into dense vector representations.
"""

import torch
import torch.nn as nn

from vasu.config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Args:
        vocab_size (int): Size of the tokenizer vocabulary.
        embedding_dim (int): Dimension of each embedding vector.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        return self.embedding(input_ids)