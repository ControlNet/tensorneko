from typing import overload, Union, Optional

import torchvision
from numpy import ndarray
from torch import Tensor

from .video_data import VideoData


class VideoWriter:
    """The VideoWriter for writing video file"""

    @staticmethod
    @overload
    def to(path: str, video: VideoData):
        """
        Write to video file from :class:`~tensorneko.io.video.video_data.VideoData`.

        Args:
            path (``str``): The path of output file.
            video (:class:`~tensorneko.io.video.video_data.VideoData`): The VideoData object for output.
        """
        ...

    @staticmethod
    @overload
    def to(path: str, video: Union[Tensor, ndarray], video_fps: float, audio: Union[Tensor, ndarray] = None,
        audio_fps: Optional[int] = None
    ):
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
        """
        ...

    @staticmethod
    def to(path: str, video: Union[Tensor, ndarray, VideoData], video_fps: float = None,
        audio: Union[Tensor, ndarray] = None, audio_fps: int = None
    ):
        """
        The implementation of :meth:`~tensorneko.io.read.video.video_writer.VideoWriter.to`.
        """
        if type(video) == VideoData:
            torchvision.io.write_video(path, video.video, fps=video.info.video_fps, audio_array=video.audio,
                audio_fps=video.info.audio_fps,
            )
        else:
            torchvision.io.write_video(path, video, video_fps, audio_fps=audio_fps, audio_array=audio)
