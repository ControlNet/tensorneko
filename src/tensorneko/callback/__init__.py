from pytorch_lightning import Callback, LightningModule, Trainer

from .lr_logger import LrLogger


class NilCallback(Callback):
    pass


class TestCallback(Callback):

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        print(trainer.training)
        print(trainer.logger_connector.callback_metrics)
