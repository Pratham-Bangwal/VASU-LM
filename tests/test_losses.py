import torch
import torch.nn.functional as F

from vasu.training.losses import language_model_loss


def test_unmasked_loss_matches_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.5, -1.0],
                [0.1, 1.4, -0.2],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 1]])

    expected = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    actual = language_model_loss(logits, targets)

    assert torch.allclose(actual, expected)


def test_masked_loss_uses_only_selected_tokens() -> None:
    logits = torch.tensor(
        [
            [
                [2.0, 0.5, -1.0],
                [0.1, 1.4, -0.2],
                [-0.4, 0.2, 1.8],
            ]
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    )

    expected = (
        per_token_loss * mask.reshape(-1)
    ).sum() / mask.sum()

    actual = language_model_loss(
        logits,
        targets,
        mask,
    )

    assert torch.allclose(actual, expected)


def test_zero_mask_produces_zero_loss_and_gradients() -> None:
    logits = torch.randn(
        2,
        3,
        5,
        requires_grad=True,
    )
    targets = torch.randint(0, 5, (2, 3))
    mask = torch.zeros(2, 3)

    loss = language_model_loss(
        logits,
        targets,
        mask,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0
