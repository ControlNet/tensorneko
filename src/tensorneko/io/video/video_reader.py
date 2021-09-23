from typing import Union

import torch
from numpy import ndarray
from torch import Tensor
from torchvision.io import read_video

from .video_data import VideoData


class VideoReader:
    """VideoReader for reading video file"""

    @staticmethod
    def of_array(video: Union[Tensor, ndarray], video_fps: float, audio: Union[Tensor, ndarray] = None,
        audio_fps: float = None, channel_last: bool = False
    ) -> VideoData:
        """
        Read video tensor from Numpy array or PyTorch Tensor directly.
        Args:
            video (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): Video input array
            video_fps (``float``): Video fps
            audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`, optional): Video input array
            audio_fps (``float``): Audio fps
            channel_last (``bool``, optional): False for (T, H, W, C) and True for (T, C, H, W)

        Returns:
            :class:`~tensorneko.io.video.video_data.VideoData`:
                A VideoData object contains a float tensor of video (T, C, H, W), with value range of 0. to 1,
                an audio tensor of (T, D) and a :class:`~tensorneko.io.video.video_data.VideoInfo` contains fps info.
        """
        if video.max() > 1:
            video = video / 255
        if channel_last:
            video = video.permute(0, 3, 1, 2)
        audio = audio or torch.tensor([]).reshape(1, 0)
        info = {
            "video_fps": video_fps,
            "audio_fps": audio_fps
        }
        return VideoData(video, audio, info)

    @staticmethod
    def of_path(path: str) -> VideoData:
        """
        Read video tensor from given file.

        Args:
            path (``str``): Path to the video file.

        Returns:
            :class:`~tensorneko.io.video.video_data.VideoData`:
                A VideoData object contains a float tensor of video (T, C, H, W), with value range of 0. to 1,
                an audio tensor of (T, D) and a :class:`~tensorneko.io.video.video_data.VideoInfo` contains fps info.
        """
        video, audio, info = read_video(path)
        video = video.permute(0, 3, 1, 2) / 255
        return VideoData(video, audio, info)

    of = of_path
