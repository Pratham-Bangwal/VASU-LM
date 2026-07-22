from pathlib import Path

from .callback import Callback


class CheckpointCallback(Callback):

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
    ):

        destination = Path("checkpoints/vasu.pt")
        if destination.resolve() == Path(trainer.config.checkpoint_path).resolve():
            return
        trainer.save_training_checkpoint(
            destination,
            epoch=epoch,
            loss=val_loss,
        )
