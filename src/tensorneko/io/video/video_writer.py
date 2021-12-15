from typing import Union, Optional

import torchvision
from numpy import ndarray
from torch import Tensor

from .video_data import VideoData
from ...util import Rearrange, dispatch


class VideoWriter:
    """The VideoWriter for writing video file"""

    @staticmethod
    @dispatch
    def to(path: str, video: VideoData, audio_codec: str = None) -> None:
        """
        Write to video file from :class:`~tensorneko.io.video.video_data.VideoData`.

        Args:
            path (``str``): The path of output file.
            video (:class:`~tensorneko.io.video.video_data.VideoData`): The VideoData object for output.
            audio_codec (``str``, optional): The audio codec if audio is required. Default: None.
        """
        if video.audio:
            if audio_codec is None:
                raise ValueError("audio_codec is required if audio is required.")

        torchvision.io.write_video(path, video.video, fps=video.info.video_fps, audio_array=video.audio,
            audio_fps=video.info.audio_fps, audio_codec=audio_codec
        )

    @staticmethod
    @dispatch
    def to(path: str, video: Union[Tensor, ndarray], video_fps: float, audio: Union[Tensor, ndarray] = None,
        audio_fps: Optional[int] = None, audio_codec: str = None
    ) -> None:
        """
        Write to video file from :class:`~torch.Tensor` or :class:`~numpy.ndarray` with (T, C, H, W).

        Args:
            path (``str``): The path of output file.
            video (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The video tensor or array with (T, C, H, W) for
                output.
            video_fps (``float``): The video fps.
            audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`, optional): The audio tensor or array with (C, T)
                for output. None means no audio in output video file. Default: None.
            audio_fps (``int``, optional): The audio fps. Default: None.
            audio_codec (``str``, optional): The audio codec if audio is required. Default: None.
        """
        if audio:
            if audio_codec is None:
                raise ValueError("audio_codec is required if audio is required.")

        rearrange = Rearrange("t c h w -> t h w c")
        torchvision.io.write_video(path, rearrange(video), video_fps, audio_fps=audio_fps, audio_array=audio)
