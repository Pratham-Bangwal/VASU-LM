from torch.utils.tensorboard import SummaryWriter

from .callback import Callback


class TensorBoardCallback(Callback):

    def __init__(self):

        self.writer = SummaryWriter("runs/vasu")

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
    ):

        self.writer.add_scalar(
            "Loss/Train",
            train_loss,
            epoch,
        )

        self.writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch,
        )

        self.writer.add_scalar(
            "Learning Rate",
            trainer.optimizer.param_groups[0]["lr"],
            epoch,
        )

    def on_train_end(self, trainer):

        self.writer.close()