from typing import Optional

from fn import F
from torch import Tensor
from torch.nn import Linear as PtLinear, Dropout

from ..neko_module import NekoModule
from ..util import ModuleFactory


class Linear(NekoModule):
    """
    An enhanced Linear version of :class:`torch.nn.Linear` with combining activation, normalization and dropout.

    Args:
        in_features (``int``): size of each input sample

        out_features  (``int``): size of each output sample

        bias (``bool``, optional): If set to ``False``, the layer will not learn an additive bias. Default: ``True``

        build_activation (``() -> torch.nn.Module``): An activation module builder to be used in the Conv2d layer.

        build_normalization (``() -> torch.nn.Module``): An normalization module builder to be used in the layer.

        normalization_after_activation (``bool``, optional): Set True then applying normalization after activation.
            Default ``False``.

        dropout_rate (``float``, optional): The dropout rate for this linear layer. 0 means no dropout applied.
            Default ``0``.

    Attributes:
        linear (:class:`~torch.nn.Linear`): The PyTorch Linear object in this layer.

        activation: (:class:`~torch.nn.Module`): The PyTorch activation module in the layer.

        normalization: (:class:`~torch.nn.Module`): The PyTorch normalization module in the layer.

        dropout: (:class:`~torch.nn.Dropout`): The PyTorch Dropout object in this layer.

    Examples::

        linear = tensorneko.layer.Linear(
            in_features=16384,
            out_features=1024,
            build_activation=torch.nn.LeakyReLU,
            build_normalization=lambda: torch.nn.BatchNorm1d(1024),
            dropout_rate=0.5
        )

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        build_activation: Optional[ModuleFactory] = None,
        build_normalization: Optional[ModuleFactory] = None,
        normalization_after_activation: bool = False,
        dropout_rate: float = 0.
    ):
        super().__init__()
        self.linear = PtLinear(in_features, out_features, bias)

        self.has_act = build_activation is not None
        if self.has_act:
            self.activation = build_activation()
        else:
            self.activation = None

        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
            self.norm_after_act = normalization_after_activation
        else:
            self.normalization = None

        self.has_dropout = dropout_rate > 0
        if self.has_dropout:
            self.dropout = Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> self.linear
        if self.has_act and self.has_norm:
            if self.norm_after_act:
                f = f >> self.activation >> self.normalization
            else:
                f = f >> self.normalization >> self.activation
        elif self.has_act and not self.has_norm:
            f = f >> self.activation
        elif not self.has_act and self.has_norm:
            f = f >> self.normalization
        if self.has_dropout:
            f = f >> self.dropout
        return f(x)
