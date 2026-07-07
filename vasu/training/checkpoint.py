import os
import torch


def save_checkpoint(model, optimizer, epoch, loss, path):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None):

    if not os.path.exists(path):
        return 0

    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}"
    )

    return checkpoint["epoch"] + 1