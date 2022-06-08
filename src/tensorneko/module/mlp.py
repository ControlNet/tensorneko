from typing import Optional, Union, Iterable, Sequence, Callable

from torch import Tensor
from torch.nn import ModuleList

from ..layer import Linear
from ..neko_module import NekoModule
from ..util import generate_inf_seq, ModuleFactory, compose


class MLP(NekoModule):
    """
    The MLP module is a sequential module with multiple linear layers.

    The activation, normalization and dropout are only applied to hidden layers.

    Args:
        neurons (``Sequence[int]``): The list of neurons for the MLP.

        bias (``bool | Iterable[bool]``, optional): The bias option of Linear layers. Default ``True``.

        build_activation (``() -> Module | Iterable[() -> Module]``, optional): The activation builder for linear
            layers. Specify single module builder will apply to all linear layers except the final one.
            Default ``None``.

        build_normalization (``() -> Module | Iterable[() -> Module]``, optional): The normalization builder for linear
            layers. Specify single module builder will apply to all linear layers except the final one.
            Default ``None``.

        normalization_after_activation (``bool``, optional): Set True then applying normalization after activation.
            Default ``False``.

        dropout_rate (``float``, optional): The dropout rate for this linear layer. 0 means no dropout applied.
            Default ``0``.

    Attributes:
        layers (:class:`~torch.nn.ModuleList`): The module list of :class:`~tensorneko.layer.Linear` layers.

    Examples:

        Create an MLP module with 4 layers, with neurons 768 -> 1024 -> 512 -> 10.
        Contains ReLU, BatchNorm and 0.5 Dropout.

        .. code-block:: python

            mlp = tensorneko.module.MLP(
                neurons=[784, 1024, 512, 10],
                build_activation=torch.nn.ReLU,
                build_normalization=[
                    lambda: torch.nn.BatchNorm1d(1024),
                    lambda: torch.nn.BatchNorm1d(512)
                ],
                dropout_rate=0.5
            )

    """

    def __init__(self, neurons: Sequence[int], bias: Union[bool, Iterable[bool]] = True,
        build_activation: Optional[Union[ModuleFactory, Iterable[ModuleFactory]]] = None,
        build_normalization: Optional[Union[ModuleFactory, Iterable[ModuleFactory]]] = None,
        normalization_after_activation: bool = False, dropout_rate: float = 0.
    ):
        super().__init__()
        n_features = neurons[1:]

        if type(bias) is bool:
            bias = generate_inf_seq([bias])

        if isinstance(build_activation, Callable) or build_activation is None:
            build_activation = generate_inf_seq([build_activation])
        if isinstance(build_normalization, Callable) or build_normalization is None:
            build_normalization = generate_inf_seq([build_normalization])

        self.layers: ModuleList[Linear] = ModuleList(
            [Linear(neurons[i], neurons[i + 1], bias[i], build_activation[i], build_normalization[i],
                normalization_after_activation, dropout_rate
            ) for i in range(len(n_features) - 1)
            ] + [
                Linear(neurons[-2], neurons[-1], bias[len(neurons) - 2])
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        f = compose(self.layers)
        return f(x)
