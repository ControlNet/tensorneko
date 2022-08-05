from pytorch_lightning import Callback, Trainer, LightningModule


class EpochNumLogger(Callback):
    """Log epoch number in each epoch start."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        key = "epoch"
        value = trainer.current_epoch
        pl_module.logger.log_metrics({key: value}, step=trainer.global_step)
        pl_module.log(key, value, logger=False, sync_dist=pl_module.distributed)
