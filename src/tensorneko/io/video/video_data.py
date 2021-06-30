from dataclasses import dataclass

from torch import Tensor


@dataclass
class VideoInfo:
    video_fps: float
    audio_fps: int


@dataclass
class VideoData:
    video: Tensor
    audio: Tensor
    info: VideoInfo

    def __init__(self, video, audio, info):
        self.video = video
        self.audio = audio
        self.info = VideoInfo(**info)
