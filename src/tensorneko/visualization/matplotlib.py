from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor


def imshow(image: Tensor, cmap=None, norm=None, aspect=None, interpolation=None,
    alpha=None, vmin=None, vmax=None, origin=None, extent=None, *,
    filternorm=True, filterrad=4.0, resample=None, url=None,
    data=None, **kwargs
):
    """
    Use matplotlib to show image of tensor with shape (C, H, W).

    Except the first parameter, other parameters are from :func:`matplotlib.pyplot.imshow`.

    Args:
        image (:class:`~torch.Tensor`): The image to show.

        cmap: The Colormap instance or registered colormap name used to map scalar data to colors.
            This parameter is ignored for RGB(A) data.

        norm: The Normalize instance used to scale scalar data to the [0, 1] range before mapping to colors using cmap.
            By default, a linear scaling mapping the lowest value to 0 and the highest to 1 is used. This parameter is
            ignored for RGB(A) data.

        aspect: The aspect ratio of the Axes. This parameter is particularly relevant for images since it determines
            whether data pixels are square.

        interpolation: The interpolation method used.

        alpha: The alpha blending value, between 0 (transparent) and 1 (opaque). If alpha is an array,
            the alpha blending values are applied pixel by pixel, and alpha must have the same shape as image.

        vmin, vmax: When using scalar data and no explicit norm, vmin and vmax define the data range that the colormap
            covers. By default, the colormap covers the complete value range of the supplied data. It is deprecated to
            use vmin/vmax when norm is given. When using RGB(A) data, parameters vmin/vmax are ignored.

        origin: Place the [0, 0] index of the array in the upper left or lower left corner of the Axes. The convention
            (the default) 'upper' is typically used for matrices and images.

        extent: The bounding box in data coordinates that the image will fill. The image is stretched individually along
            x and y to fill the box.

        filternorm: A parameter for the antigrain image resize filter (see the antigrain documentation). If filternorm
            is set, the filter normalizes integer values and corrects the rounding errors. It doesn't do anything with
            the source floating point values, it corrects only integers according to the rule of 1.0 which means that
            any sum of pixel weights must be equal to 1.0. So, the filter function must produce a graph of the proper
            shape.

        filterrad: The filter radius for filters that have a radius parameter, i.e. when interpolation is one
            of: 'sinc', 'lanczos' or 'blackman'.
        resample: When True, use a full resampling method. When False, only resample when the output image is larger
            than the input image.

        url: Set the url of the created AxesImage

        **kwargs: Other parameters for :func:`matplotlib.pyplot.imshow`.

    """
    image = rearrange(image, "c h w -> h w c").cpu()
    plt.imshow(image, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent,
        filternorm=filternorm, filterrad=filterrad, resample=resample, url=url, data=data, **kwargs)
    plt.show()
