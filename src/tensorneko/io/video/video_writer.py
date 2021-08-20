from typing import overload, Union

import torchvision
from numpy import ndarray
from torch import Tensor

from .video_data import VideoData


class VideoWriter:

    @staticmethod
    @overload
    def to(path: str, video: VideoData):
        ...

    @staticmethod
    @overload
    def to(path: str, video: Union[Tensor, ndarray], video_fps: float, audio: Union[Tensor, ndarray], audio_fps: int):
        ...

    @staticmethod
    def to(path: str, video: Union[Tensor, ndarray, VideoData], video_fps: float = None,
        audio: Union[Tensor, ndarray] = None, audio_fps: int = None
    ):
        if type(video) == VideoData:
            torchvision.io.write_video(path, video.video, fps=video.info.video_fps, audio_array=video.audio,
                audio_fps=video.info.audio_fps,
            )
        else:
            torchvision.io.write_video(path, video, video_fps, audio_fps=audio_fps, audio_array=audio)
