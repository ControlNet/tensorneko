from lightning.pytorch import Callback, Trainer, LightningModule


class DisplayMetricsCallback(Callback):
    """A test callback to display the callback metrics."""

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        print(trainer.callback_metrics)
