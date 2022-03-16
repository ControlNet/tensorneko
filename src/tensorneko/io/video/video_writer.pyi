from typing import Optional, overload, Union

from numpy import ndarray
from torch import Tensor

from tensorneko.io.video import VideoData


class VideoWriter:
    @staticmethod
    @overload
    def to(path: str, video: VideoData, audio_codec: Optional[str] = None) -> None: ...

    @staticmethod
    @overload
    def to(path: str, video: Union[Tensor, ndarray], video_fps: float, audio: Union[Tensor, ndarray] = None,
        audio_fps: Optional[int] = None, audio_codec: Optional[str] = None) -> None: ...
