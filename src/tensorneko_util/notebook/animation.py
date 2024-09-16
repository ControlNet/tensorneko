from typing import Sequence, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def to_animation_html(frames: Union[Sequence[np.ndarray], np.ndarray], interval: int = 100) -> str:
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0])
    def update_frame(i):
        im.set_array(frames[i])
        return [im]

    return FuncAnimation(fig, update_frame, frames=len(frames), interval=interval).to_jshtml()


def show_animation(frames: Union[Sequence[np.ndarray], np.ndarray], interval: int = 100):
    return HTML(to_animation_html(frames, interval))
