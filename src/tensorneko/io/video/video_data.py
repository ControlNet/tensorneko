from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class VideoInfo:
    """A dataclass for info of video"""
    video_fps: float
    audio_fps: Optional[int]


@dataclass
class VideoData:
    """
    A dataclass for output of :class:`~.video_reader.VideoReader`

    Attributes:
        video (:class:`~torch.Tensor`): video float tensor of (T, C, H, W)
        audio (:class:`~torch.Tensor`): audio float tensor of (C, T)
        info (:class:`VideoInfo`): video and audio fps info
    """
    video: Tensor
    audio: Tensor
    info: VideoInfo

    def __init__(self, video, audio, info):
        self.video = video
        self.audio = audio
        info_args = info if len(info) == 2 else {
            "video_fps": info["video_fps"],
            "audio_fps": None
        }
        self.info = VideoInfo(**info_args)
