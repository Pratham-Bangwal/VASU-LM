import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU Feed Forward Network.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):

        return self.w3(
            self.silu(self.w1(x)) * self.w2(x)
        )