from typing import Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from .util import summarize_dict_by, Shape


class Model(LightningModule):

    def __init__(self, name: str,
        input_shape: Optional[Shape] = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.can_plot_graph = input_shape is not None
        if self.can_plot_graph:
            self.input_shape = input_shape
            self.example_input_array = torch.rand([1, *input_shape])
        self.history = []

    @property
    def logger(self) -> Optional[LightningLoggerBase]:
        if not self.trainer:
            return None
        elif self.training:
            return self.trainer.logger_train
        else:
            return self.trainer.logger_val

    # history logger
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if self.trainer.log_on_epoch and self.logger is not None:
            self.log_on_training_epoch_end(outputs)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if self.logger is not None:
            self.log_on_validation_epoch_end(outputs)

    def training_step_end(self, output: STEP_OUTPUT) -> STEP_OUTPUT:
        if self.trainer.log_on_step \
            and self.logger is not None \
            and (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.log_on_training_step_end(output)
        return output

    def log_on_training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        history_item = {}
        for key in outputs[0].keys():
            getter = summarize_dict_by(key, torch.mean)
            value = getter(outputs)
            history_item[key] = value
            self.log(key, value, on_epoch=True, on_step=False, logger=True)
        self.history.append(history_item)

    def log_on_validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if len(self.history) == 0 or "loss" not in self.history[-1].keys():
            return

        for key in outputs[0].keys():
            getter = summarize_dict_by(key, torch.mean)
            value = getter(outputs)
            self.history[-1]["val_" + key] = value
            self.log(key, value, on_epoch=True, on_step=False)
            self.log(f"val_{key}", value, on_epoch=True, on_step=False, logger=False, prog_bar=True)

    def log_on_training_step_end(self, output: STEP_OUTPUT) -> None:
        history_item = {}
        for key, value in output.items():
            history_item[key] = value
            self.log(key, value, on_epoch=False, on_step=True, logger=True)
        self.history.append(history_item)
