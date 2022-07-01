import pathlib
from typing import Optional, Union

import numpy as np
from einops import rearrange

from .video_data import VideoData
from .._default_backends import _default_video_io_backend
from ...backend.visual_lib import VisualLib
from ...util import dispatch
from ...util.type import T_ARRAY


class VideoWriter:
    """The VideoWriter for writing video file"""

    @classmethod
    @dispatch
    def to(cls, path: str, video: VideoData, audio_codec: str = None, channel_first: bool = False,
        backend: VisualLib = None
    ) -> None:
        """
        Write to video file from :class:`~tensorneko.io.video.video_data.VideoData`.

        Args:
            path (``str``): The path of output file.
            video (:class:`~tensorneko.io.video.video_data.VideoData`): The VideoData object for output.
            audio_codec (``str``, optional): The audio codec if audio is required. Default: None.
            channel_first (``bool``, optional): Get image dimension (T, H, W, C) if False or (T, C, H, W) if True.
                Default: False.
            backend (:class:`~tensorneko.backend.visual_lib.VisualLib`, optional): VisualLib backend.
                Default: opencv if installed, then torchvision if installed, then ffmpeg if available.
        """
        cls.to(path, video.video, video.info.video_fps, video.audio, video.info.audio_fps, audio_codec, channel_first, backend)

    @classmethod
    @dispatch
    def to(cls, path: str, video: T_ARRAY, video_fps: float, audio: T_ARRAY = None,
        audio_fps: Optional[int] = None, audio_codec: str = None, channel_first: bool = False,
        backend: VisualLib = None
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
            channel_first (``bool``, optional): Get image dimension (T, H, W, C) if False or (T, C, H, W) if True.
                Default: False.
            backend (:class:`~tensorneko.backend.visual_lib.VisualLib`, optional): VisualLib backend.
                Default: opencv if installed, then torchvision if installed, then ffmpeg if available.
        """
        if audio:
            if audio_codec is None:
                raise ValueError("audio_codec is required if audio is required.")
        backend = backend or _default_video_io_backend()
        if channel_first:
            video = rearrange(video, "t c h w -> t h w c")

        # to uint8
        if isinstance(video, np.ndarray):
            video = (video * 255).astype(np.uint8)
        else:
            try:
                import torch
                if isinstance(video, torch.Tensor):
                    if backend == VisualLib.PYTORCH:
                        video = (video * 255).type(torch.IntTensor)
                    else:
                        video = (video.numpy() * 255).astype(np.uint8)
                else:
                    raise ValueError("Unknown data type. The image array type must be numpy.ndarray or torch.Tensor.")
            except ImportError:
                raise ValueError("Unknown data type. The image array type must be numpy.ndarray or torch.Tensor.")

        ext = pathlib.Path(path).suffix

        if backend == VisualLib.OPENCV:
            if audio is not None:
                raise ValueError("Write audio is not supported in opencv backend.")
            if not VisualLib.opencv_available():
                raise ValueError("Opencv is not installed.")
            import cv2
            if ext == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            elif ext == ".mp4":
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                raise NotImplementedError("Only .avi and .mp4 are supported in opencv backend.")
            writer = cv2.VideoWriter(path, fourcc, video_fps, (video.shape[2], video.shape[1]))
            for i in range(video.shape[0]):
                writer.write(cv2.cvtColor(video[i], cv2.COLOR_RGB2BGR))
            writer.release()
        elif backend == VisualLib.PYTORCH:
            if not VisualLib.pytorch_available():
                raise ValueError("Torchvision is not installed.")
            import torchvision
            torchvision.io.write_video(path, rearrange(video), video_fps, audio_fps=audio_fps, audio_array=audio)
        elif backend == VisualLib.FFMPEG:
            if audio is not None:
                raise ValueError("Write audio is not supported in ffmpeg backend.")
            if not VisualLib.ffmpeg_available():
                raise ValueError("FFMPEG is not installed.")
            import ffmpeg
            pipe = ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{video.shape[2]}x{video.shape[1]}") \
                .output(path, format=ext[1:], pix_fmt="rgb24", r=video_fps) \
                .overwrite_output() \
                .run_async(pipe_stdin=True)

            for i in range(video.shape[0]):
                pipe.stdin.write(video[i].tobytes())

            pipe.stdin.close()
            pipe.wait()
        else:
            raise ValueError("Unknown backend. Should be one of VisualLib.OPENCV, VisualLib.PYTORCH, VisualLib.FFMPEG.")

    def __new__(cls, path: str, video: Union[T_ARRAY, VideoData], *args, **kwargs):
        """Alias of :func:`~tensorneko.io.video.video_io.to`."""
        return cls.to(path, video, *args, **kwargs)
