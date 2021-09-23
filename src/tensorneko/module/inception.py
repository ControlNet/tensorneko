from __future__ import annotations

from fn import F
from torch import Tensor
from torch.nn import ModuleList, Sequential, Module

from ..neko_module import NekoModule
from ..layer import Concatenate


class InceptionModule(NekoModule):
    """
    InceptionModule is the module has multiple route for input feature and then concat the outputs of all routes. This
    is introduced firstly by Szegedy, et al. (2015).

    Attributes:
        sub_seqs (:class:`torch.nn.ModuleList`): The module list of sub sequential modules.

    References:
        Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper
        with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).

    """

    def __init__(self):
        super().__init__()
        self.sub_seqs = ModuleList()
        self.channel_concat = Concatenate(dim=1)

    def add_sub_sequence(self, *layers: Module) -> InceptionModule:
        """
        Add sub sequential module for this InceptionModule.

        Args:
            *layers (:class:`~torch.nn.Module`): The var length parameter layers are the modules used in one route.

        Returns:
            :class:`InceptionModule`: The self InceptionModule object.

        Examples:

            Add a route with 1x1Conv -> 3x3Conv layers for this module.

            .. code-block:: python

                inception_module.add_sub_sequence(
                    tensorneko.layer.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        build_activation=torch.nn.ReLU,
                        build_normalization=lambda: torch.nn.BatchNorm2d(64),
                    ),
                    tensorneko.layer.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=(3, 3),
                        padding=(1, 1),
                        build_activation=torch.nn.ReLU,
                        build_normalization=lambda: torch.nn.BatchNorm2d(64),
                    ),
                )

        """
        self.sub_seqs.append(Sequential(*layers))
        return self

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> (map, lambda seq: seq(x)) >> list >> self.channel_concat
        return f(self.sub_seqs)

