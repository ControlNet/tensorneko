import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, GELU, ELU, ReLU, CrossEntropyLoss, L1Loss, MSELoss, BCELoss


# Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
# https://arxiv.org/abs/1908.08681v1
# implemented for PyTorch / FastAI by lessw2020
# github: https://github.com/lessw2020/mish
# TODO: replace to PyTorch implementation in PyTorch 1.9
class _Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


class StringGetter:
    activation_mapping = {
        "LEAKYRELU": LeakyReLU,
        "GELU": GELU,
        "ELU": ELU,
        "RELU": ReLU,
        "MISH": _Mish,
    }
    loss_mapping = {
        "CROSSENTROPYLOSS": CrossEntropyLoss,
        "L1LOSS": L1Loss,
        "MSELOSS": MSELoss,
        "BCELOSS": BCELoss
    }

    mapping = {
        "activation": activation_mapping,
        "loss": loss_mapping
    }

    def __init__(self, type: str):
        self.type = type
        self.mapping = StringGetter.mapping[self.type]

    def __call__(self, name: str):
        return self.get(name)

    def get(self, name: str):
        return self.mapping[name.upper()]


activation_getter = StringGetter("activation")
loss_getter = StringGetter("loss")
