import torch
import torch.nn.functional as F


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
        Computes the autoregressive language modeling loss.

        Args:
            logits: (B, T, V)
            targets: (B, T)

        Returns:
            Cross-entropy loss.
    """

    B, T, V = logits.shape

    logits = logits.view(B * T, V)
    targets = targets.view(B * T)

    return F.cross_entropy(logits, targets)