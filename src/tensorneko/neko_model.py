from abc import abstractmethod
from typing import Optional, Dict, Union, Sequence, Any, List

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor

from .neko_module import NekoModule
from .neko_trainer import NekoTrainer
from .util import summarize_dict_by, Shape

EPOCH_OUTPUT = List[STEP_OUTPUT]


class NekoModel(LightningModule, NekoModule):
    """
    An abstract class for models. In this module, the loss and other metrics will be automatically logged in
    TensorBoard.

    Args:
        name (``str``): Model name

        input_shape (:class:`~tensorneko.util.Shape`, optional):
            An optional argument can allow it to plot a graph for TensorBoard

        *args: Other arguments for :class:`~lightning.pytorch.core.lightning.LightningModule`

        **kwargs: Other arguments for :class:`~lightning.pytorch.core.lightning.LightningModule`
    """
    trainer: NekoTrainer

    def __init__(self, name: str,
        input_shape: Optional[Shape] = None, distributed=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.distributed = distributed
        self.can_plot_graph = input_shape is not None
        if self.can_plot_graph:
            self.input_shape = input_shape
            self.example_input_array = torch.rand([1, *input_shape])
        self.history = []
        self.training_step_outputs = []
        self.validation_step_outputs = []

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def training_step(self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        The method inherit from :meth:`~lightning.pytorch.core.lightning.LightningModule.training_step`.

        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.

            batch_idx (``int``): Integer displaying index of this batch

        Returns:
            ``Dict`` [``str``, :class:`~torch.Tensor`]:
                A dictionary. Can include any keys, but must include the key ``'loss'``

        Notes:
            The return value can be other types, but you need to handle them by override the
                method :meth:`~lightning.pytorch.TrainingModule.training_step_end`

        Examples::

            def training_step(self, batch, batch_idx):
                x, y = batch
                out = self(x)
                loss = self.loss(out, x)
                return {"loss": loss}
        """
        ...

    @abstractmethod
    def validation_step(self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        The method inherit from :meth:`~lightning.pytorch.core.lightning.LightningModule.validation_step`.

        Operates on a single batch of data from the validation set.
        In this step you'd might generate examples or calculate anything of interest like accuracy.

        Args:
            batch (:class:`~torch.Tensor` | (:class:`~torch.Tensor`, ...) | [:class:`~torch.Tensor`, ...]):
                The output of your :class:`~torch.utils.data.DataLoader`. A tensor, tuple or list.

            batch_idx (``int``): The index of this batch

            dataloader_idx (``int``, optional): The index of the dataloader that produced this batch
                (only if multiple val dataloaders used)

        Returns:
            ``Dict`` [``str``, :class:`~torch.Tensor`]:
                A dictionary. Can include any keys, but must include the key ``'loss'``

        Examples::

            def validation_step(self, batch, batch_idx):
                x, y = batch
                out = self(x)
                loss = self.loss(out, x)
                acc = self.acc(out, x)
                return {"loss": loss, "acc": acc}
        """
        ...

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """
        Inherit from :meth:`~lightning.pytorch.core.lightning.LightningModule.predict_step`.

        Step function called during :meth:`~lightning.pytorch.trainer.trainer.Trainer.predict`.
        By default, it calls :meth:`~lightning.pytorch.core.lightning.LightningModule.forward`.
        Override to add any processing logic.

        Args:
            batch (:class:`~torch.Tensor`): Current batch
            batch_idx (``int``): Index of current batch
            dataloader_idx (``int``, optional): Index of the current dataloader

        Return:
            ``Any``: The predicted output
        """
        return self.forward(batch)

    def test_step(self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Dict[str, Tensor]:
        return self.validation_step(batch, batch_idx, dataloader_idx)

    @abstractmethod
    def configure_optimizers(self):
        """Inherit from :meth:`~lightning.pytorch.core.lightning.LightningModule.configure_optimizers`."""
        ...

    @property
    def logger(self) -> Optional[Logger]:
        """
        Returns:
            :class:`~lightning.pytorch.loggers.LightningLoggerBase` | ``None``:
                The training or validation logger for the module.
        """
        if not self.trainer:
            return None
        else:
            return self.trainer.logger

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        """For each training step end, log the metrics. Append the outputs of training step to the list"""
        outputs = {k: v.detach() for k, v in outputs.items()}
        self.training_step_outputs.append(outputs)
        if self.trainer.log_on_step \
            and self.logger is not None \
            and self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            self.log_on_training_step_end(outputs)
        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Append the outputs of validation step to the list"""
        outputs = {k: v.detach() for k, v in outputs.items()}
        self.validation_step_outputs.append(outputs)
        super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_train_epoch_end(self) -> None:
        """For each training epoch end, log the metrics"""
        if self.trainer.log_on_epoch and self.logger is not None:
            self.log_on_training_epoch_end(self.training_step_outputs)
        self.training_step_outputs.clear()
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """For each validation epoch end, log the metrics"""
        if self.logger is not None:
            self.log_on_validation_epoch_end(self.validation_step_outputs)
        self.validation_step_outputs.clear()
        super().on_validation_epoch_end()

    def log_on_training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Log the training epoch outputs"""
        history_item = {}
        for key in outputs[0].keys():
            getter = summarize_dict_by(key, torch.mean)
            value = getter(outputs)
            history_item[key] = value
            if self.logger is not None:
                self.logger.log_metrics({key: value}, step=self.trainer.global_step)
            self.log(key, value, on_epoch=True, on_step=False, logger=False, sync_dist=self.distributed)
        self.history.append(history_item)

    def log_on_validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        """Log the validation epoch outputs"""
        if len(self.history) == 0 or "loss" not in self.history[-1].keys():
            return

        for key in outputs[0].keys():
            getter = summarize_dict_by(key, torch.mean)
            value = getter(outputs)
            self.history[-1]["val_" + key.split("/")[-1]] = value
            if self.logger is not None:
                self.logger.log_metrics({key: value}, step=self.trainer.global_step)
            self.log(key, value, on_epoch=True, on_step=False, logger=False, sync_dist=self.distributed)
            self.log(f"val_{key.split('/')[-1]}", value, on_epoch=True, on_step=False, logger=False,
                prog_bar=True, sync_dist=self.distributed)

    def log_on_training_step_end(self, output: STEP_OUTPUT) -> None:
        """Log the training step outputs"""
        history_item = {}
        for key, value in output.items():
            history_item[key] = value
            if self.logger is not None:
                self.logger.log_metrics({key: value}, step=self.trainer.global_step)
            self.log(key, value, on_epoch=False, on_step=True, logger=False, prog_bar=key == "loss",
                sync_dist=self.distributed)
        self.history.append(history_item)

    def on_test_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """For each test step end, log the metrics"""
        self.log_dict(outputs, sync_dist=self.distributed)
        super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def log_image(self, name: str, image: torch.Tensor) -> None:
        """Log an image to the logger"""
        if self.logger is None:
            return

        self.logger.experiment.add_image(name, torch.clip(image, 0, 1), self.trainer.global_step)

    def log_histogram(self, name: str, values: torch.Tensor) -> None:
        """Log a histogram to the logger"""
        if self.logger is None:
            return

        self.logger.experiment.add_histogram(name, values, self.trainer.global_step)

    def on_train_start(self) -> None:
        """Log the model graph to tensorboard when the input shape is set"""
        if self.can_plot_graph and self.logger is not None:
            self.logger.log_graph(self, self.example_input_array)
        super().on_train_start()
