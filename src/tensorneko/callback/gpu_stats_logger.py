from lightning.pytorch import Callback, Trainer, LightningModule


class GpuStatsLogger(Callback):
    """Log GPU stats for each training epoch"""

    def __init__(self, delay: float = 0.5):
        try:
            from gpumonitor.monitor import GPUStatMonitor
        except ImportError:
            raise ImportError("gpumonitor is required to use GPUStatsLogger")

        self.monitor = GPUStatMonitor(delay=delay)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.monitor.reset()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for gpu in self.monitor.average_stats.gpus:
            logged_info = {
                f"gpu{gpu.index}_memory_used": gpu.memory_used / 1024,
                f"gpu{gpu.index}_memory_total": gpu.memory_total / 1024,
                f"gpu{gpu.index}_memory_util": gpu.memory_used / gpu.memory_total,
                f"gpu{gpu.index}_temperature": float(gpu.temperature),
                f"gpu{gpu.index}_utilization": gpu.utilization / 100,
                f"gpu{gpu.index}_power_draw": float(gpu.power_draw),
                f"gpu{gpu.index}_power_percentage": gpu.power_draw / gpu.power_limit,
                f"gpu{gpu.index}_fan_speed": float(gpu.fan_speed),
            }
            pl_module.logger.log_metrics(logged_info, step=trainer.global_step)
            pl_module.log_dict(logged_info, logger=False, sync_dist=pl_module.distributed)
