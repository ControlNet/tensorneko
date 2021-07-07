from typing import Dict

from torch.nn import LeakyReLU, GELU, ELU, ReLU, CrossEntropyLoss, L1Loss, MSELoss, BCELoss, BCEWithLogitsLoss, Mish, \
    Module


class StringGetter:
    """Util class for retrieve module class from string"""

    _activation_mapping: Dict[str, Module] = {
        "LEAKYRELU": LeakyReLU,
        "GELU": GELU,
        "ELU": ELU,
        "RELU": ReLU,
        "MISH": Mish,
    }

    _loss_mapping: Dict[str, Module] = {
        "CROSSENTROPYLOSS": CrossEntropyLoss,
        "L1LOSS": L1Loss,
        "MSELOSS": MSELoss,
        "BCELOSS": BCELoss,
        "BCEWITHLOGITSLOSS": BCEWithLogitsLoss
    }

    _mapping: Dict[str, Dict[str, Module]] = {
        "activation": _activation_mapping,
        "loss": _loss_mapping
    }

    def __init__(self, type: str):
        self.type = type
        self.dict = StringGetter._mapping[self.type]

    def __call__(self, name: str):
        return self._get(name)

    def _get(self, name: str):
        return self.dict[name.upper()]


get_activation = StringGetter("activation")
"""The function for getting activation function from its name"""


get_loss = StringGetter("loss")
"""The function for getting loss function from its name"""
