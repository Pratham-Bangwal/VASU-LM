import torch
import torch.nn.functional as F


def language_model_loss(
    logits,
    targets,
    mask=None,
):
    B, T, V = logits.shape

    logits = logits.view(
        B * T,
        V,
    )

    targets = targets.view(
        B * T,
    )

    loss = F.cross_entropy(
        logits,
        targets,
        reduction="none",
    )

    if mask is not None:
        mask = mask.view(-1)
        loss = loss * mask
        return loss.sum() / (
            mask.sum() + 1e-8
        )

    return loss.mean()