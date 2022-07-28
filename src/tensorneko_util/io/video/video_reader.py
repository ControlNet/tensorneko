from typing import Optional

import numpy as np
from einops import rearrange

from .video_data import VideoData
from .._default_backends import _default_video_io_backend
from ...backend.visual_lib import VisualLib


class VideoReader:

    @classmethod
    def of(cls, path: str, channel_first: bool = True, backend: Optional[VisualLib] = None) -> VideoData:
        """
        Read video array from given file.

        Args:
            path (``str``): Path to the video file.
            channel_first (``bool``, optional): Get image dimension (T, H, W, C) if False or (T, C, H, W) if True.
                Default: True.
            backend (:class:`~tensorneko.backend.visual_lib.VisualLib`, optional): VisualLib backend.
                Default: opencv if installed, then torchvision if installed, then ffmpeg if available.

        Returns:
            :class:`~tensorneko.io.video.video_data.VideoData`:
                A VideoData object contains a float tensor of video, with value range of 0. to 1,
                an audio tensor of (T, C) and a :class:`~tensorneko.io.video.video_data.VideoInfo` contains fps info.
        """
        backend = backend or _default_video_io_backend()
        if backend == VisualLib.OPENCV:
            if not VisualLib.opencv_available():
                raise ValueError("OpenCV is not installed.")
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("Failed to open video file: {}".format(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            video = np.stack(frames, axis=0)
            if channel_first:
                video = rearrange(video, 'T H W C -> T C H W')
            return VideoData(video, None, {"video_fps": fps})
        elif backend == VisualLib.PYTORCH:
            if not VisualLib.pytorch_available():
                raise ValueError("Torchvision is not installed.")
            import torchvision
            video, audio, info = torchvision.io.read_video(path)
            if channel_first:
                video = rearrange(video, 'T H W C -> T C H W')
            audio = rearrange(audio, 'C T -> T C')
            return VideoData(video, audio, info)
        elif backend == VisualLib.FFMPEG:
            if not VisualLib.ffmpeg_available():
                raise ValueError("FFMPEG is not installed.")
            import ffmpeg

            probe = ffmpeg.probe(path)
            for each in probe['streams']:
                if each['codec_type'] == 'video':
                    fps = each['avg_frame_rate']
                    height = each['height']
                    width = each['width']
                    break
            else:
                raise ValueError("Failed to get video from {}".format(path))

            out, _ = (
                ffmpeg
                .input(path)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True)
            )
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            if channel_first:
                video = rearrange(video, 'T H W C -> T C H W')
            return VideoData(video, None, {"video_fps": fps})
        else:
            raise ValueError("Unknown backend: {}".format(backend))

    def __new__(cls, path: str, channel_first: bool = False, backend: Optional[VisualLib] = None) -> VideoData:
        """Alias of :meth:`~.VideoReader.of`"""
        return cls.of(path, channel_first, backend)
