from time import time

from pytorch_lightning import Callback, Trainer, LightningModule


class EpochTimeLogger(Callback):
    """Log spent time for each training epoch"""
    start_time: float

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.start_time = time()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        elapsed_time = time() - self.start_time
        pl_module.logger.log_metrics({"epoch_time": elapsed_time}, step=trainer.global_step)
        pl_module.log("epoch_time", elapsed_time, logger=False, sync_dist=pl_module.distributed)

