import torch.nn.functional as F


def language_model_loss(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T)
    """

    B, T, V = logits.shape

    logits = logits.view(B * T, V)
    targets = targets.view(B * T)

    return F.cross_entropy(logits, targets)