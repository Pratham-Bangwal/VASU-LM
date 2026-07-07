import torch
import torch.nn as nn

from vasu.config import ModelConfig


class SwiGLU(nn.Module):
    """
    SwiGLU Feed Forward Network.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.dim, config.hidden_dim, bias=config.bias)
        self.w3 = nn.Linear(config.hidden_dim, config.dim, bias=config.bias)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):

        return self.w3(
            self.silu(self.w1(x)) * self.w2(x)
        )