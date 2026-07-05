"""
Root Mean Square Layer Normalization (RMSNorm)

Reference:
https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn

from vasu.config import ModelConfig


class RMSNorm(nn.Module):
    """
    RMSNorm used in modern LLMs such as Llama, Mistral,
    Gemma and DeepSeek.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * rms)