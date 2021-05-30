from pytorch_lightning import Callback, Trainer, LightningModule


class LrLogger(Callback):
    """Log learning rate in each epoch start."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for i, optimizer in enumerate(trainer.optimizers):
            for j, params in enumerate(optimizer.param_groups):
                pl_module.log(f"opt{i}_lr{j}", params["lr"])
