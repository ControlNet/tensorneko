from typing import Any

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class SystemStatsLogger(Callback):
    """Log system stats for each training epoch"""

    def __init__(self, on_epoch: bool = True, on_step: bool = False):
        try:
            import psutil
        except ImportError:
            raise ImportError("psutil is required to use SystemStatsLogger")
        self.psutil = psutil
        self.on_epoch = on_epoch
        self.on_step = on_step
        assert self.on_epoch or self.on_step, "on_epoch and on_step cannot be both False"

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.on_epoch:
            return
        cpu_usage = self.psutil.cpu_percent()
        memory_usage = self.psutil.virtual_memory().percent
        logged_info = {
            "system/cpu_usage_epoch": cpu_usage,
            "system/memory_usage_epoch": memory_usage
        }
        pl_module.logger.log_metrics(logged_info, step=trainer.global_step)
        pl_module.log_dict(logged_info, logger=False, sync_dist=pl_module.distributed)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if not self.on_step:
            return
        cpu_usage = self.psutil.cpu_percent()
        memory_usage = self.psutil.virtual_memory().percent
        logged_info = {
            "system/cpu_usage_step": cpu_usage,
            "system/memory_usage_step": memory_usage
        }
        pl_module.logger.log_metrics(logged_info, step=trainer.global_step)
        pl_module.log_dict(logged_info, logger=False, sync_dist=pl_module.distributed)
