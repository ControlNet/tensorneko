from typing import Union, Optional

from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Conv2d, Conv3d

from ..neko_module import NekoModule
from ..util import ModuleFactory, Shape, F


class Patching(NekoModule):
    """
    Patching images or videos.

    If the input is an image (B, C, H, W), the output is a tensor of shape (B, C, N_H, N_W, P_H, P_W).
    If the input is a video (B, C, T, H, W), the output is a tensor of shape (B, C, N_T, N_H, N_W, P_T, P_H, P_W).
    """

    def __init__(self, patch_size: Union[int, Shape]):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class PatchEmbedding2d(NekoModule):
    """
    The patch embedding layer from Vision Transformer (ViT) (Dosovitskiy, et al., 2020). This layer will take an image
    as input, and divide to several patches, and project to a provided number of channels.

    Here it is implemented by an equivalent convolution layer with same stride length and kernel size.

    Args:
        input_size (:class:`~tensorneko.util.type.Shape`): The size of input shape. (C, H, W).

        patch_size (``int`` | :class:`~tensorneko.util.type.Shape`): The side length of patch or a patch Shape of
            (H, W).

        embedding (``int``): Output embedding dimension.

        strides (``int`` | :class:`~tensorneko.util.type.Shape`, optional): The strides of the patch window.
            Default is the size of patch.

        build_normalization (``() -> torch.nn.Module``, optional): An normalization module builder to be used in the
            layer. Default ``None``.

    Examples::

        patch_emb = tensorneko.layer.PatchEmbedding2d(
            input_size=(3, 224, 224),
            patch_size=(16, 16),
            embedding=768
        )

        print(patch_emb(rand(8, 3, 224, 224)).shape)  # (8, 196, 768)

    References:
        Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N.
        (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
        arXiv preprint arXiv:2010.11929.

    """

    def __init__(self, input_size: Shape, patch_size: Union[int, Shape], embedding: int,
        strides: Optional[Union[int, Shape]] = None,
        build_normalization: Optional[ModuleFactory] = None
    ):
        super().__init__()
        # channel, height, width
        c, h, w = input_size
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


class PatchEmbedding3d(NekoModule):
    """
    The 3D cube version of Patch Embedding layer.

    Args:
        input_size (:class:`~tensorneko.util.type.Shape`): The size of input shape. (C, T, H, W).
        patch_size (``int`` | :class:`~tensorneko.util.type.Shape`): The side length of patch or a patch Shape of
            (T, H, W).
        embedding (``int``): Output embedding dimension.
        strides (``int`` | :class:`~tensorneko.util.type.Shape`, optional): The strides of the patch window.
            Default is the size of patch.
        build_normalization (``() -> torch.nn.Module``, optional): An normalization module builder to be used in the
            layer. Default ``None``.

    Examples::

        patch_emb = tensorneko.layer.PatchEmbedding3d(
            input_size=(3, 16, 224, 224),
            patch_size=(2, 16, 16),
            embedding=1024
        )

        print(patch_emb(rand(8, 3, 16, 224, 224)).shape)  # (8, 1568, 1024)

    """

    def __init__(self, input_size: Shape, patch_size: Union[int, Shape], embedding: int,
        strides: Optional[Union[int, Shape]] = None,
        build_normalization: Optional[ModuleFactory] = None
    ):
        super().__init__()
        # channel, time, height, width
        c, t, h, w = input_size
        # patch_time, patch_height, patch_width
        pt, ph, pw = (patch_size, patch_size, patch_size) if type(patch_size) is int else patch_size

        # configure the strides for conv3d
        if strides is None:
            # no specified means no overlap and gap between patches
            strides = (pt, ph, pw)
        elif type(strides) is int:
            # transform the side length of strides to 3D
            strides = (strides, strides, strides)

        self.projection = Conv3d(c, embedding, kernel_size=(pt, ph, pw), stride=strides)
        self.has_norm = build_normalization is not None
        if self.has_norm:
            self.normalization = build_normalization()
        self.rearrange = Rearrange("b d nt nh nw -> b (nt nh nw) d")

    def forward(self, x: Tensor) -> Tensor:
        f = F() >> self.projection >> self.rearrange
        if self.has_norm:
            f = f >> self.normalization
        return f(x)
