class Callback:

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(
        self,
        trainer,
        epoch,
        train_loss,
        val_loss,
    ):
        pass