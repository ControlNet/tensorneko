from typing import Iterable

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT

from .util import summarize_dict_by


class Model(LightningModule):

    def __init__(self, name: str, input_shape: Iterable[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.example_input_array = torch.rand([1, *input_shape])
        self.history = []

    @property
    def logger(self):
        if not self.trainer:
            return None
        elif self.training:
            return self.trainer.logger_train
        else:
            return self.trainer.logger_val

    # history logger
    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        history_item = {}
        for key in outputs[0].keys():
            getter = summarize_dict_by(key, torch.mean)
            value = getter(outputs)
            history_item[key] = value
            self.log(key, value, on_epoch=True, on_step=False, logger=True)
        self.history.append(history_item)

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        if len(self.history) == 0 or "loss" not in self.history[-1].keys():
            return

        for key in outputs[0].keys():
            getter = summarize_dict_by(key, torch.mean)
            value = getter(outputs)
            self.history[-1]["val_" + key] = value
            self.log(key, value, on_epoch=True, on_step=False)
            self.log(f"val_{key}", value, on_epoch=True, on_step=False, logger=False, prog_bar=True)