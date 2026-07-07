import torch
import torch.nn as nn
from vasu.config import TrainConfig


def build_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )