from typing import Optional, Union, Iterable, Sequence, Callable

from torch.nn import Module, ModuleList

from ..layer import Linear
from ..util.func import generate_inf_seq
from ..util.type import ModuleFactory


class MLP(Module):

    def __init__(self, neurons: Sequence[int], bias: bool = True,
        build_activation: Optional[Union[ModuleFactory, Iterable[ModuleFactory]]] = None,
        build_normalization: Optional[Union[ModuleFactory, Iterable[ModuleFactory]]] = None,
        normalization_after_activation: bool = False
    ):
        super().__init__()
        n_features = neurons[1:]

        if isinstance(build_activation, Callable) or build_activation is None:
            build_activation = generate_inf_seq([build_activation])
        if isinstance(build_normalization, Callable) or build_normalization is None:
            build_normalization = generate_inf_seq([build_normalization])

        self.layers: ModuleList[Linear] = ModuleList(
            [Linear(neurons[i], neurons[i + 1], bias, build_activation[i], build_normalization[i],
                normalization_after_activation
            ) for i in range(len(n_features) - 1)
            ] + [
                Linear(neurons[-2], neurons[-1], bias)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
