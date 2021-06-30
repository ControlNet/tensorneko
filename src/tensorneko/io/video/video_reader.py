from einops import rearrange
from torchvision.io import read_video

from tensorneko.io.video.video_data import VideoData


class VideoReader:

    @staticmethod
    def of(path: str) -> VideoData:
        video, audio, info = read_video(path)
        video = video.permute(0, 3, 1, 2) / 255
        return VideoData(video, audio, info)
