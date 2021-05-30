from pytorch_lightning import Callback, LightningModule, Trainer

from .lr_logger import LrLogger

__all__ = [
    "NilCallback",
    "DisplayMetricsCallback",
    "LrLogger"
]


class NilCallback(Callback):
    """A Nil Callback doing nothing in training."""
    pass


class DisplayMetricsCallback(Callback):
    """A test callback to display the callback metrics."""
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(trainer.logger_connector.callback_metrics)
