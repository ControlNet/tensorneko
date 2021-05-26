from einops import rearrange
from torchvision.io import read_video

from tensorneko.io.video.video_data import VideoData


class VideoReader:
    @staticmethod
    def of(path: str) -> VideoData:
        video, audio, info = read_video(path)
        video = rearrange(video, "t h w c -> t c h w") / 255
        return VideoData(video, audio, info)