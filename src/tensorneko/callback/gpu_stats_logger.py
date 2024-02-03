from typing import Any

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT


class GpuStatsLogger(Callback):
    """Log GPU stats for each training epoch"""

    def __init__(self, delay: float = 0.5, on_epoch: bool = True, on_step: bool = False):
        try:
            from gpumonitor.monitor import GPUStatMonitor
        except ImportError:
            raise ImportError("gpumonitor is required to use GPUStatsLogger")

        self.monitor_epoch = GPUStatMonitor(delay=delay) if on_epoch else None
        self.monitor_step = GPUStatMonitor(delay=delay) if on_step else None
        self.on_epoch = on_epoch
        self.on_step = on_step
        assert self.on_epoch or self.on_step, "on_epoch and on_step cannot be both False"

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.on_epoch:
            return
        self.monitor_epoch.reset()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.on_epoch:
            return
        for gpu in self.monitor_epoch.average_stats.gpus:
            logged_info = {
                f"system/gpu{gpu.index}_memory_used_epoch": gpu.memory_used / 1024,
                f"system/gpu{gpu.index}_memory_total_epoch": gpu.memory_total / 1024,
                f"system/gpu{gpu.index}_memory_util_epoch": gpu.memory_used / gpu.memory_total,
                f"system/gpu{gpu.index}_temperature_epoch": float(gpu.temperature),
                f"system/gpu{gpu.index}_utilization_epoch": gpu.utilization / 100,
                f"system/gpu{gpu.index}_power_draw_epoch": float(gpu.power_draw),
                f"system/gpu{gpu.index}_power_percentage_epoch": gpu.power_draw / gpu.power_limit,
                f"system/gpu{gpu.index}_fan_speed_epoch": float(gpu.fan_speed) if gpu.fan_speed is not None else 0.,
            }
            pl_module.logger.log_metrics(logged_info, step=trainer.global_step)
            pl_module.log_dict(logged_info, logger=False, sync_dist=pl_module.distributed)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        if not self.on_step:
            return
        self.monitor_step.reset()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if not self.on_step:
            return
        for gpu in self.monitor_step.average_stats.gpus:
            logged_info = {
                f"system/gpu{gpu.index}_memory_used_step": gpu.memory_used / 1024,
                f"system/gpu{gpu.index}_memory_total_step": gpu.memory_total / 1024,
                f"system/gpu{gpu.index}_memory_util_step": gpu.memory_used / gpu.memory_total,
                f"system/gpu{gpu.index}_temperature_step": float(gpu.temperature),
                f"system/gpu{gpu.index}_utilization_step": gpu.utilization / 100,
                f"system/gpu{gpu.index}_power_draw_step": float(gpu.power_draw),
                f"system/gpu{gpu.index}_power_percentage_step": gpu.power_draw / gpu.power_limit,
                f"system/gpu{gpu.index}_fan_speed_step": float(gpu.fan_speed) if gpu.fan_speed is not None else 0.,
            }
            pl_module.logger.log_metrics(logged_info, step=trainer.global_step)
            pl_module.log_dict(logged_info, logger=False, sync_dist=pl_module.distributed)
