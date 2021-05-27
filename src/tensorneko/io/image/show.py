from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor


def imshow(image: Tensor, cmap=None, norm=None, aspect=None, interpolation=None,
    alpha=None, vmin=None, vmax=None, origin=None, extent=None, *,
    filternorm=True, filterrad=4.0, resample=None, url=None,
    data=None, **kwargs
):
    image = rearrange(image, "c h w -> h w c").cpu()
    plt.imshow(image, cmap, norm, aspect, interpolation, alpha, vmin, vmax, origin, extent,
        filternorm=filternorm, filterrad=filterrad, resample=resample, url=url, data=data, **kwargs)
    plt.show()
