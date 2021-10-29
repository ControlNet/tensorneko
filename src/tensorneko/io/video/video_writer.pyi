from types import Union
from typing import Optional, overload

from numpy import ndarray
from torch import Tensor

from tensorneko.io.video import VideoData


class VideoWriter:
    @staticmethod
    @overload
    def to(path: str, video: VideoData) -> None: ...

    @staticmethod
    @overload
    def to(path: str, video: Union[Tensor, ndarray], video_fps: float, audio: Union[Tensor, ndarray] = None,
        audio_fps: Optional[int] = None) -> None: ...
