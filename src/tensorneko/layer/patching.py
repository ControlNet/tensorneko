from typing import Union, Iterator, Optional

from einops.layers.torch import Rearrange
from fn import F
from torch import Tensor
from torch.nn import Module, Conv2d

from ..util import ModuleFactory


class Patching(Module):

    def __init__(self):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return x


class PatchEmbedding2d(Module):

    def __init__(self, image_size: Iterator[int], patch_size: Union[int, Iterator[int]], embedding: int,
        strides: Optional[Union[int, Iterator[int]]] = None,
        build_normalization: Optional[ModuleFactory] = None
    ):
        super().__init__()
        # height, width and channel
        c, h, w = image_size
        # patch_height, patch_width
        ph, pw = (patch_size, patch_size) if type(patch_size) is int else patch_size

        # configure the strides for conv2d
        if strides is None:
            # no specified means no overlap and gap between patches
            strides = (ph, pw)
        elif type(strides) is int:
            # transform the side length of strides to 2D
            strides = (strides, strides)

        self.projection = Conv2d(c, embedding, kernel_size=(ph, pw), stride=strides)
        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
        self.rearrange = Rearrange("b d nh nw -> b (nh nw) d")


    def forward(self, x: Tensor) -> Tensor:
        f = F() >> self.projection >> self.rearrange
        if self.has_norm:
            f = f >> self.normalization
        return f(x)
