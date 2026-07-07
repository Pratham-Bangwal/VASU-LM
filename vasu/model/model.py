"""
VASU-1 Language Model
"""

import torch
import torch.nn as nn

from vasu.config import ModelConfig

from .block import TransformerBlock
from .embedding import TokenEmbedding
from .rmsnorm import RMSNorm


class VASUModel(nn.Module):
    """
    Decoder-only Transformer Language Model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.embedding = TokenEmbedding(config)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.norm = RMSNorm(config.dim)

        self.lm_head = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = self.embedding.embedding.weight

    def forward(
            self,
            input_ids:torch.Tensor,
        ) -> torch.Tensor:

        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, sequence_length)

        Returns:
            Logits of shape (batch_size, sequence_length, vocab_size)
        """

        x = self.embedding(input_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = x / (self.config.dim ** 0.5)   

        logits = self.lm_head(x)

        return logits   
    
    