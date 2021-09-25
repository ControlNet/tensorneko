from typing import Dict

from torch.nn import LeakyReLU, GELU, ELU, ReLU, Mish, Module, ReLU6, PReLU, SELU, Sigmoid, \
    Tanh, Softplus, Softshrink, Softsign, Softmin, Softmax, LogSoftmax

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, BCELoss, BCEWithLogitsLoss, NLLLoss, \
    NLLLoss2d, PoissonNLLLoss, KLDivLoss, HingeEmbeddingLoss, MarginRankingLoss, \
    MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, SoftMarginLoss, \
    TripletMarginLoss, CosineEmbeddingLoss, HuberLoss, SmoothL1Loss


class StringGetter:
    """Util class for retrieve module class from string"""

    _activation_mapping: Dict[str, Module] = {
        "LEAKYRELU": LeakyReLU,
        "GELU": GELU,
        "ELU": ELU,
        "RELU": ReLU,
        "RELU6": ReLU6,
        "PRELU": PReLU,
        "MISH": Mish,
        "SELU": SELU,
        "TANH": Tanh,
        "SIGMOID": Sigmoid,
        "SOFTPLUS": Softplus,
        "SOFTSHRINK": Softshrink,
        "SOFTSIGN": Softsign,
        "LOGSOFTMAX": LogSoftmax,
        "SOFTMIN": Softmin,
        "SOFTMAX": Softmax
    }

    _loss_mapping: Dict[str, Module] = {
        "CROSSENTROPYLOSS": CrossEntropyLoss,
        "L1LOSS": L1Loss,
        "MSELOSS": MSELoss,
        "BCELOSS": BCELoss,
        "BCEWITHLOGITSLOSS": BCEWithLogitsLoss,
        "NLLLOSS": NLLLoss,
        "NLLLOSS2D": NLLLoss2d,
        "POISSONNLLLOSS": PoissonNLLLoss,
        "KLDIVLOSS": KLDivLoss,
        "HINGEEMBEDDINGLOSS": HingeEmbeddingLoss,
        "MARGINRANKINGLOSS": MarginRankingLoss,
        "MULTIMARGINLOSS": MultiMarginLoss,
        "MULTILABELSOFTMARGINLOSS": MultiLabelSoftMarginLoss,
        "MULTILABELMARGINLOSS": MultiLabelMarginLoss,
        "SOFTMARGINLOSS": SoftMarginLoss,
        "TRIPLETMARGINLOSS": TripletMarginLoss,
        "COSINEEMBEDDINGLOSS": CosineEmbeddingLoss,
        "HUBERLOSS": HuberLoss,
        "SMOOTHL1LOSS": SmoothL1Loss
    }

    _mapping: Dict[str, Dict[str, Module]] = {
        "activation": _activation_mapping,
        "loss": _loss_mapping
    }

    def __init__(self, type_: str):
        self.type = type_
        self.dict = StringGetter._mapping[self.type]

    def __call__(self, name: str):
        return self._get(name)

    def _get(self, name: str):
        return self.dict[name.upper()]


get_activation = StringGetter("activation")
"""The function for getting activation function from its name"""


get_loss = StringGetter("loss")
"""The function for getting loss function from its name"""
