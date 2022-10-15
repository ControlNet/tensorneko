from typing import Union, Optional

from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import Conv2d, Conv3d

from ..neko_module import NekoModule
from ..util import ModuleFactory, Shape, F


class Patching(NekoModule):
    """
    Patching input tensor with specified patch size.

    If the `patch_size` is tuple of dims, it will patch last same number of dims. E.g. if `patch_size=(16, 16)`, and the
        input tensor has shape `(b, c, h, w)`, it will patch `(b, c, h // 16, w // 16, 16, 16)`.

    If the `patch_size` is int, it will patch from dim 2 to last. E.g. if `patch_size=16`, and the input tensor has
        shape `(a, b, c, d, e, f)`, it will patch `(a, b, c // 16, d // 16, e // 16, f // 16)`.

    Args:
        patch_size (``int`` | :class:`~tensorneko.util.type.Shape`): The patch size.

    Examples::

        import torch
        x = torch.rand(2, 3, 224, 224)
        patching = Patching(patch_size=16)

        print(patching(x).shape)  # (2, 3, 14, 14, 16, 16)

    """

    def __init__(self, patch_size: Union[int, Shape]):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        x_dim = x.dim()

        assert x_dim > 2, "The input must be at least 3-dimensional."
        if type(self.patch_size) is int:
            patch_dims = range(2, x_dim)
            self.patch_size = (self.patch_size,) * len(patch_dims)
        else:
            patch_dims = range(x_dim - len(self.patch_size), x_dim)

        for i, patch_dim in enumerate(patch_dims):
            x = x.unfold(patch_dim, self.patch_size[i], self.patch_size[i])

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
        x = self.projection(x)
        x = self.rearrange(x)
        if self.has_norm:
            x = self.normalization(x)
        return x


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
        x = self.projection(x)
        x = self.rearrange(x)
        if self.has_norm:
            x = self.normalization(x)
        return x
