import torch

from vasu.config import TrainConfig


def build_optimizer(model, config: TrainConfig):

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )