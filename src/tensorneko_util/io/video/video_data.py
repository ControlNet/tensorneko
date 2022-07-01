from dataclasses import dataclass
from typing import Optional

from ...util.type import T_ARRAY


@dataclass
class VideoInfo:
    """A dataclass for info of video"""
    video_fps: float
    audio_fps: Optional[int]

    def __iter__(self):
        return iter((self.video_fps, self.audio_fps))


@dataclass
class VideoData:
    """
    A dataclass for output of :class:`~.video_reader.VideoReader`

    Attributes:
        video (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): video float tensor of (T, C, H, W)
        audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): audio float tensor of (C, T)
        info (:class:`VideoInfo`): video and audio fps info
    """
    video: T_ARRAY
    audio: T_ARRAY
    info: VideoInfo

    def __init__(self, video, audio, info):
        self.video = video
        self.audio = audio
        info_args = info if len(info) == 2 else {
            "video_fps": info["video_fps"],
            "audio_fps": None
        }
        self.info = VideoInfo(**info_args)

    def __iter__(self):
        return iter((self.video, self.audio, self.info))
