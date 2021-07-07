from typing import Union, Optional

from einops.layers.torch import Rearrange
from fn import F
from torch import Tensor
from torch.nn import Conv2d

from ..neko_module import NekoModule
from ..util import ModuleFactory, Shape


class Patching(NekoModule):

    def __init__(self):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return x


class PatchEmbedding2d(NekoModule):
    """
    The patch embedding layer from Vision Transformer (ViT) (Dosovitskiy, et al., 2020). This layer will take a image
    as input, and divide to patches, and project to a provided number of channels.

    Here it is implemented by a equivalent convolution layer with same stride length and kernel size.

    Args:
        image_size (:class:`~tensorneko.util.type.Shape`): The size of input shape. (C, H, W).

        patch_size (``int`` | :class:`~tensorneko.util.type.Shape`): The side length of patch or a patch Shape of
            (H, W).

        embedding (``int``): Output embedding dimension.

        strides (``int`` | :class:`~tensorneko.util.type.Shape`, optional): The strides of the patch window.
            Default is the size of patch.

        build_normalization (``() -> torch.nn.Module``, optional): An normalization module builder to be used in the
            layer. Default ``None``.

    Examples::

        patch_emb = tensorneko.layer.PatchEmbedding2d(
            image_size=(3, 256, 256),
            patch_size=(16, 16),
            embedding=1024
        )

    References:
        Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N.
        (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
        arXiv preprint arXiv:2010.11929.

    """

    def __init__(self, image_size: Shape, patch_size: Union[int, Shape], embedding: int,
        strides: Optional[Union[int, Shape]] = None,
        build_normalization: Optional[ModuleFactory] = None
    ):
        super().__init__()
        # channel, height, width
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
