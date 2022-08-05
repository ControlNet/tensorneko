import os
from datetime import timedelta
from pathlib import Path
from time import time
from typing import Optional, Union, List, Dict

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import ModelCheckpoint, Checkpoint
from pytorch_lightning.loggers import Logger, TensorBoardLogger
from pytorch_lightning.plugins import PLUGIN_INPUT
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import Strategy
from pytorch_lightning.trainer.connectors.accelerator_connector import _LITERAL_WARN

from .callback import NilCallback, LrLogger, EpochNumLogger, EpochTimeLogger


class NekoTrainer(Trainer):

    def __init__(self,
        logger: Optional[str],
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        log_every_n_steps: int = 50,
        gpus: int = 0,
        precision: Union[int, str] = 32,
        default_root_dir: Optional[str] = None,
        gradient_clip_val: float = 0.0,
        gradient_clip_algorithm: str = 'norm',
        num_nodes: int = 1,
        num_processes: Optional[int] = None,  # TODO: Remove in 2.0
        tpu_cores: Optional[Union[List[int], str, int]] = None,  # TODO: Remove in 2.0
        ipus: Optional[int] = None,  # TODO: Remove in 2.0
        devices: Optional[Union[List[int], str, int]] = None,
        auto_select_gpus: bool = False,
        enable_progress_bar: bool = True,
        overfit_batches: Union[int, float] = 0.0,
        track_grad_norm: Union[int, float, str] = -1,
        check_val_every_n_epoch: Optional[int] = 1,
        fast_dev_run: Union[int, bool] = False,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        limit_train_batches: Optional[Union[int, float]] = None,
        limit_val_batches: Optional[Union[int, float]] = None,
        limit_test_batches: Optional[Union[int, float]] = None,
        limit_predict_batches: Optional[Union[int, float]] = None,
        val_check_interval: Optional[Union[int, float]] = None,
        accelerator: Optional[Union[str, Accelerator]] = None,
        strategy: Optional[Union[str, Strategy]] = None,
        sync_batchnorm: bool = False,
        enable_model_summary: bool = True,
        weights_save_path: Optional[str] = None,  # TODO: Remove in 1.8
        num_sanity_val_steps: int = 2,
        resume_from_checkpoint: Optional[Union[Path, str]] = None,
        profiler: Optional[Union[Profiler, str]] = None,
        benchmark: Optional[bool] = None,
        deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
        reload_dataloaders_every_n_epochs: int = 0,
        auto_lr_find: Union[bool, str] = False,
        replace_sampler_ddp: bool = True,
        detect_anomaly: bool = False,
        auto_scale_batch_size: Union[str, bool] = False,
        plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
        amp_backend: str = "native",
        amp_level: Optional[str] = None,
        move_metrics_to_cpu: bool = False,
        multiple_trainloader_mode: str = "max_size_cycle",
    ):
        super().__init__()

        # define a random logger name
        self.log_name = f"{logger}_{int(time())}"
        if callbacks is None:
            callbacks = []
        # build checkpoint callback or from user defined
        if enable_checkpointing and len([c for c in callbacks if isinstance(c, Checkpoint)]) == 0:
            # use default checkpoint callback
            new_callback = ModelCheckpoint(
                dirpath=os.path.join("logs", self.log_name, "checkpoints"),
                save_last=True, filename="{epoch}-{val_loss:.3f}", monitor="val_loss",
                mode="min"
            )
        elif enable_checkpointing and len([c for c in callbacks if isinstance(c, Checkpoint)]) > 0:
            # use user defined checkpoint callback
            new_callback = NilCallback()
        elif not enable_checkpointing and len([c for c in callbacks if isinstance(c, Checkpoint)]) > 0:
            # raise error because checkpoint callback is not allowed
            raise ValueError("Checkpoint callback is not allowed when checkpointing is disabled.")
        else:
            # if no checkpoint callback is defined, skip
            new_callback = NilCallback()

        # merge the checkpoint callback with other callbacks
        if callbacks is None:
            cbs = [new_callback]
        elif isinstance(callbacks, Callback):
            cbs = [callbacks, new_callback]
        else:
            cbs = callbacks + [new_callback]

        # set learning rate logger
        cbs.extend([LrLogger(), EpochNumLogger(), EpochTimeLogger()])

        # setup the log mode
        self.log_every_n_steps = log_every_n_steps
        self.log_on_epoch = log_every_n_steps == 0
        self.log_on_step = log_every_n_steps > 0
        if self.log_on_epoch:
            log_every_n_steps = 10000000

        super().__init__(
            logger=logger is not None,
            enable_checkpointing=enable_checkpointing,
            callbacks=cbs,
            default_root_dir=default_root_dir,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            num_nodes=num_nodes,
            num_processes=num_processes,
            devices=devices,
            gpus=gpus,
            auto_select_gpus=auto_select_gpus,
            tpu_cores=tpu_cores,
            ipus=ipus,
            enable_progress_bar=enable_progress_bar,
            overfit_batches=overfit_batches,
            track_grad_norm=track_grad_norm,
            check_val_every_n_epoch=check_val_every_n_epoch,
            fast_dev_run=fast_dev_run,
            accumulate_grad_batches=accumulate_grad_batches,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            max_time=max_time,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            strategy=strategy,
            sync_batchnorm=sync_batchnorm,
            precision=precision,
            enable_model_summary=enable_model_summary,
            weights_save_path=weights_save_path,
            num_sanity_val_steps=num_sanity_val_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            profiler=profiler,
            benchmark=benchmark,
            deterministic=deterministic,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
            auto_lr_find=auto_lr_find,
            replace_sampler_ddp=replace_sampler_ddp,
            detect_anomaly=detect_anomaly,
            auto_scale_batch_size=auto_scale_batch_size,
            plugins=plugins,
            amp_backend=amp_backend,
            amp_level=amp_level,
            move_metrics_to_cpu=move_metrics_to_cpu,
            multiple_trainloader_mode=multiple_trainloader_mode
        )
        # init loggers for training and validation
        self.has_no_logger = logger is None

        self.logger_train = TensorBoardLogger(save_dir=self.default_root_dir, name="logs",
                                              version=os.path.join(self.log_name, "train"), log_graph=False
                                              # TODO: Fix log_Graph
                                              ) if self.has_no_logger is not None else None
        self.logger_val = TensorBoardLogger(save_dir=self.default_root_dir, name="logs",
                                            version=os.path.join(self.log_name, "val"), log_graph=False
                                            ) if self.has_no_logger is not None else None
        self._loggers = []

    @property
    def logger(self) -> Optional[Logger]:
        if self.has_no_logger:
            return None
        if self.training:
            return self.logger_train
        else:
            return self.logger_val
