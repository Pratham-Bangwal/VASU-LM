from .callback import Callback
from vasu.training.sample import generate_sample


class SampleGenerationCallback(Callback):

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
    ):
        generate_sample(
            trainer.model,
            trainer.tokenizer,
            trainer.device,
        )