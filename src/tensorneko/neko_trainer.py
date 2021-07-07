from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Union, Optional, List, Dict

from fn import F
from pytorch_lightning import Callback
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import Plugin
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.profiler import BaseProfiler

from .callback import LrLogger
from .callback.nil_callback import NilCallback


def NekoTrainer(
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
) -> _NekoTrainer:
    """
    The NekoTrainer wrapper follows the NekoTrainer in PyTorch Lightning, but with some modification in logger, checkpoint,
    and callbacks.

    Two TensorBoard are used for training and validation respectively, for better visual effects in TensorBoard
    visualization. For example, you can see training loss and validation loss in a same line chart. The TensorBoard
    logs are saved in `default_root_dir`/logs.

    Also, the learning rate for each optimizer are logged.

    Args:
        logger (``bool``): Option if you want to use TensorBoard logger in the trainer.

        checkpoint_callback (``bool`` | :class:`~pytorch_lightning.Callback`):
            Set `True` for default checkpoint callback,
            set `False` for no checkpoint callback,
            or a user-defined checkpoint callback object.

        callbacks (``List`` [:class:`~pytorch_lightning.Callback`] | :class:`~pytorch_lightning.Callback`):
            Add a callback or list of callbacks.

        default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
            Default: ``os.getcwd()``.
            Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

        gradient_clip_val: 0 means don't clip.

        gradient_clip_algorithm: 'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm'

        process_position: orders the progress bar when running multiple models on same machine.

        num_nodes: number of GPU nodes for distributed training.

        num_processes: number of processes for distributed training with distributed_backend="ddp_cpu"

        gpus: number of gpus to train on (int) or which GPUs to train on (list or str) applied per node

        auto_select_gpus: If enabled and `gpus` is an integer, pick available
            gpus automatically. This is especially useful when
            GPUs are configured to be in "exclusive mode", such
            that only one process at a time can access them.

        tpu_cores: How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

        log_gpu_memory: None, 'min_max', 'all'. Might slow performance

        progress_bar_refresh_rate: How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
            Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
            a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).

        overfit_batches: Overfit a fraction of training data (float) or a set number of batches (int).

        track_grad_norm: -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

        check_val_every_n_epoch: Check val every n train epochs.

        fast_dev_run: runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
            of train, val and test to find any bugs (ie: a sort of unit test).

        accumulate_grad_batches: Accumulates grads every k batches or as set up in the dict.

        max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
            If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.

        min_epochs: Force training for at least these many epochs. Disabled by default (None).
            If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.

        max_steps: Stop training after this number of steps. Disabled by default (None).

        min_steps: Force training for at least these number of steps. Disabled by default (None).

        max_time: Stop training after this amount of time has passed. Disabled by default (None).
            The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
            :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
            :class:`datetime.timedelta`.

        limit_train_batches: How much of training dataset to check (float = fraction, int = num_batches)

        limit_val_batches: How much of validation dataset to check (float = fraction, int = num_batches)

        limit_test_batches: How much of test dataset to check (float = fraction, int = num_batches)

        limit_predict_batches: How much of prediction dataset to check (float = fraction, int = num_batches)

        val_check_interval: How often to check the validation set. Use float to check within a training epoch,
            use int to check every n steps (batches).

        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

        log_every_n_steps: Default 0. How often to log within steps (defaults to every 50 steps).
            0 means only log per epoch.

        accelerator: Previously known as distributed_backend (dp, ddp, ddp2, etc...).
            Can also take in an accelerator object for custom hardware.

        sync_batchnorm: Synchronize batch norm layers between process groups/whole world.

        precision: Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or
            TPUs.

        weights_summary: Prints a summary of the weights when training begins.

        weights_save_path: Where to save weights if specified. Will override default_root_dir
            for checkpoints only. Use this if for whatever reason you need the checkpoints
            stored in a different place than the logs written in `default_root_dir`.
            Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            Defaults to `default_root_dir`.

        num_sanity_val_steps: Sanity check runs n validation batches before starting the training routine.
            Set it to `-1` to run all batches in all validation dataloaders.

        truncated_bptt_steps: Deprecated in v1.3 to be removed in 1.5.
            Please use :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` instead.

        resume_from_checkpoint: Path/URL of the checkpoint from which training is resumed. If there is
            no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
            training will start from the beginning of the next epoch.

        profiler: To profile individual steps during training and assist in identifying bottlenecks.

        benchmark: If true enables cudnn.benchmark.

        deterministic: If true enables cudnn.deterministic.

        reload_dataloaders_every_epoch: Set to True to reload dataloaders every epoch.

        auto_lr_find: If set to True, will make trainer.tune() run a learning rate finder,
            trying to optimize initial learning for faster convergence. trainer.tune() method will
            set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
            To use a different key set a string instead of True with the key name.

        replace_sampler_ddp: Explicitly enables or disables sampler replacement. If not specified this
            will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
            train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
            you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.

        terminate_on_nan: If set to True, will terminate training (by raising a `ValueError`) at the
            end of each training batch, if any of the parameters or the loss are NaN or +/-inf.

        auto_scale_batch_size: If set to True, will `initially` run a batch size
            finder trying to find the largest batch size that fits into memory.
            The result will be stored in self.batch_size in the LightningModule.
            Additionally, can be set to either `power` that estimates the batch size through
            a power search or `binsearch` that estimates the batch size through a binary search.

        prepare_data_per_node: If True, each LOCAL_RANK=0 will call prepare data.
            Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data

        plugins: Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.

        amp_backend: The mixed precision backend to use ("native" or "apex")

        amp_level: The optimization level to use (O1, O2, etc...).

        distributed_backend: deprecated. Please use 'accelerator'

        move_metrics_to_cpu: Whether to force internal logged metrics to be moved to cpu.
            This can save some gpu memory, but can make training slower. Use with attention.

        multiple_trainloader_mode: How to loop over the datasets when there are multiple train loaders.
            In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
            and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
            reload when reaching the minimum length of datasets.

        stochastic_weight_avg: Whether to use `Stochastic Weight Averaging (SWA)
            <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_`

    Returns:
        :class:`~._NekoTrainer`: A NekoTrainer object.

    """

    build_trainer = F(_NekoTrainer, logger, checkpoint_callback, callbacks, default_root_dir, gradient_clip_val,
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
        _NekoTrainer.logger = None
        obj = build_trainer()
        _NekoTrainer.logger = _logger
    except AttributeError:
        obj = build_trainer()
        _NekoTrainer.logger = _logger
    return obj


@property
def _logger(self):
    if self.has_no_logger:
        return None
    if self.training:
        return self.logger_train
    else:
        return self.logger_val


class _NekoTrainer(Trainer):
    """
    The NekoTrainer wrapper follows the NekoTrainer in PyTorch Lightning, but with some modification in logger, checkpoint,
    and callbacks.

    Please use :func:`NekoTrainer` to instantiate a NekoTrainer object.

    Two TensorBoard are used for training and validation respectively, for better visual effects in TensorBoard
    visualization. For example, you can see training loss and validation loss in a same line chart. The TensorBoard
    logs are saved in `default_root_dir`/logs.

    Also, the learning rate for each optimizer are logged.
    """

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
        log_every_n_steps: int = 0,  # 0 means for log_every_epoch
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
        # define a random logger name
        self.log_name = f"{logger}_{int(time())}"
        # build checkpoint callback or from user defined
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

        # merge the checkpoint callback with other callbacks
        if callbacks is None:
            cbs = [checkpoint_cb_obj]
        elif isinstance(callbacks, Callback):
            cbs = [callbacks, checkpoint_cb_obj]
        else:
            cbs = callbacks + [checkpoint_cb_obj]

        # set learning rate logger
        cbs.append(LrLogger())

        # setup the log mode
        self.log_every_n_steps = log_every_n_steps
        self.log_on_epoch = log_every_n_steps == 0
        self.log_on_step = log_every_n_steps > 0
        if self.log_on_epoch:
            log_every_n_steps = 10000000

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
        # init loggers for training and validation
        self.has_no_logger = logger is None

        self.logger_train = TensorBoardLogger(save_dir=self.default_root_dir, name="logs",
            version=os.path.join(self.log_name, "train"), log_graph=False  # TODO: Fix log_Graph
        ) if self.has_no_logger is not None else None
        self.logger_val = TensorBoardLogger(save_dir=self.default_root_dir, name="logs",
            version=os.path.join(self.log_name, "val"), log_graph=False
        ) if self.has_no_logger is not None else None
