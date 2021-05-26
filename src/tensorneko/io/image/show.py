from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor


def imshow(image: Tensor):
    image = rearrange(image, "c h w -> h w c")
    plt.imshow(image)
    plt.show()
