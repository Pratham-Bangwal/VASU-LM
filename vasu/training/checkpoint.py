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


def load_checkpoint(model, optimizer, path):

    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint