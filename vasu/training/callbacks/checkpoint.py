import torch

from .callback import Callback


class CheckpointCallback(Callback):

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
    ):

        torch.save(
            {
                "model": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": trainer.best_val_loss,
            },
            "checkpoints/vasu.pt",
        )