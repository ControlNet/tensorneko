import os
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Union, Optional, List, Dict

from fn import F
from pytorch_lightning import Callback
from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profiler import BaseProfiler

from .callback import NilCallback, LrLogger


class Trainer(PLTrainer):

    def __init__(
        self,
        logger: Optional[str],
        checkpoint_callback: Union[bool, Callback] = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = 'norm',
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: Optional[int] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 0,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = 'top',
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[
            Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        distributed_backend: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = 'max_size_cycle',
        stochastic_weight_avg: bool = False
    ):
        self.log_name = f"{logger}_{int(time())}"
        if type(checkpoint_callback) is bool and checkpoint_callback is True:
            checkpoint_cb_obj = ModelCheckpoint(dirpath=os.path.join("logs", self.log_name, "checkpoints"),
                save_last=True, filename="{epoch}-{val_loss:.3f}", monitor="val_loss", mode="min"
            )
            checkpoint_cb_bool = True
        elif type(checkpoint_callback) is ModelCheckpoint:
            checkpoint_cb_obj = checkpoint_callback
            checkpoint_cb_bool = True
        elif type(checkpoint_callback) is bool and checkpoint_callback is False:
            checkpoint_cb_obj = NilCallback()
            checkpoint_cb_bool = False
        else:
            raise ValueError("Not available value of checkpoint_callback.")

        if callbacks is None:
            cbs = [checkpoint_cb_obj]
        elif isinstance(callbacks, Callback):
            cbs = [callbacks, checkpoint_cb_obj]
        else:
            cbs = callbacks + [checkpoint_cb_obj]

        cbs.append(LrLogger())

        self.log_every_n_steps = log_every_n_steps
        self.log_on_epoch = log_every_n_steps == 0
        self.log_on_step = log_every_n_steps > 0
        if self.log_on_epoch:
            log_every_n_steps = 1000000

        super().__init__(logger is not None, checkpoint_cb_bool, cbs, default_root_dir, gradient_clip_val,
            gradient_clip_algorithm, process_position, num_nodes, num_processes, gpus, auto_select_gpus,
            tpu_cores, log_gpu_memory, progress_bar_refresh_rate, overfit_batches, track_grad_norm,
            check_val_every_n_epoch, fast_dev_run, accumulate_grad_batches, max_epochs, min_epochs,
            max_steps, min_steps, max_time, limit_train_batches, limit_val_batches, limit_test_batches,
            limit_predict_batches, val_check_interval, flush_logs_every_n_steps, log_every_n_steps,
            accelerator, sync_batchnorm, precision, weights_summary, weights_save_path,
            num_sanity_val_steps, truncated_bptt_steps, resume_from_checkpoint, profiler, benchmark,
            deterministic, reload_dataloaders_every_epoch, auto_lr_find, replace_sampler_ddp,
            terminate_on_nan, auto_scale_batch_size, prepare_data_per_node, plugins, amp_backend,
            amp_level, distributed_backend, move_metrics_to_cpu, multiple_trainloader_mode,
            stochastic_weight_avg
        )
        self.has_no_logger = logger is None

        self.logger_train = TensorBoardLogger(save_dir=self.default_root_dir, name="logs",
            version=os.path.join(self.log_name, "train"), log_graph=False  # TODO: Fix log_Graph
        ) if self.has_no_logger is not None else None
        self.logger_val = TensorBoardLogger(save_dir=self.default_root_dir, name="logs",
            version=os.path.join(self.log_name, "val"), log_graph=False
        ) if self.has_no_logger is not None else None

    @staticmethod
    def build(
        logger: Optional[str],
        checkpoint_callback: Union[bool, Callback] = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = 'norm',
        process_position: int = 0,
        num_nodes: int = 1,
        num_processes: int = 1,
        gpus: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        tpu_cores: Optional[Union[List[int], str, int]] = None,
        log_gpu_memory: Optional[str] = None,
        progress_bar_refresh_rate: Optional[int] = None,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: int = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Union[int, Dict[int, int], List[list]] = 1,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        limit_predict_batches: Union[int, float] = 1.0,
        val_check_interval: Union[int, float] = 1.0,
        flush_logs_every_n_steps: int = 100,
        log_every_n_steps: int = 0,
        accelerator: Optional[Union[str, Accelerator]] = None,
        sync_batchnorm: bool = False,
        precision: int = 32,
        weights_summary: Optional[str] = 'top',
        weights_save_path: Optional[str] = None,
        num_sanity_val_steps: int = 2,
        truncated_bptt_steps: Optional[int] = None,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[BaseProfiler, str]] = None,
        benchmark: bool = False,
        deterministic: bool = False,
        reload_dataloaders_every_epoch: bool = False,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        terminate_on_nan: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        prepare_data_per_node: bool = True,
        plugins: Optional[Union[List[Union[Plugin, ClusterEnvironment, str]], Plugin, ClusterEnvironment, str]] = None,
        amp_backend: str = 'native',
        amp_level: str = 'O2',
        distributed_backend: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = 'max_size_cycle',
        stochastic_weight_avg: bool = False
    ):

        build_trainer = F(Trainer, logger, checkpoint_callback, callbacks, default_root_dir, gradient_clip_val,
            gradient_clip_algorithm, process_position, num_nodes, num_processes, gpus, auto_select_gpus,
            tpu_cores, log_gpu_memory, progress_bar_refresh_rate, overfit_batches, track_grad_norm,
            check_val_every_n_epoch, fast_dev_run, accumulate_grad_batches, max_epochs, min_epochs,
            max_steps, min_steps, max_time, limit_train_batches, limit_val_batches, limit_test_batches,
            limit_predict_batches, val_check_interval, flush_logs_every_n_steps, log_every_n_steps,
            accelerator, sync_batchnorm, precision, weights_summary, weights_save_path,
            num_sanity_val_steps, truncated_bptt_steps, resume_from_checkpoint, profiler, benchmark,
            deterministic, reload_dataloaders_every_epoch, auto_lr_find, replace_sampler_ddp,
            terminate_on_nan, auto_scale_batch_size, prepare_data_per_node, plugins, amp_backend,
            amp_level, distributed_backend, move_metrics_to_cpu, multiple_trainloader_mode,
            stochastic_weight_avg
        )
        try:
            Trainer.logger = None
            obj = build_trainer()
            Trainer.logger = _logger
        except AttributeError:
            obj = build_trainer()
            Trainer.logger = _logger
        return obj


@property
def _logger(self):
    if self.has_no_logger:
        return None
    if self.training:
        return self.logger_train
    else:
        return self.logger_val
