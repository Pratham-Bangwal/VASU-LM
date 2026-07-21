import torch

from vasu.model import RMSNorm


def test_rmsnorm_preserves_shape():
    x = torch.randn(4, 16, 24)
    norm = RMSNorm(24)

    y = norm(x)

    assert y.shape == x.shape
