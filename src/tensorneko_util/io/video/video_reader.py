from itertools import islice, takewhile
from typing import Optional

import numpy as np
from einops import rearrange

from .video_data import VideoData
from .._default_backends import _default_video_io_backend
from ...backend.visual_lib import VisualLib
from ...util import dispatch


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

            out, _ = ffmpeg.input(path).output('pipe:', format='rawvideo', pix_fmt='rgb24').run(quiet=True)
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            if channel_first:
                video = rearrange(video, 'T H W C -> T C H W')
            return VideoData(video, None, {"video_fps": fps})
        else:
            raise ValueError("Unknown backend: {}".format(backend))

    @classmethod
    def with_indexes(cls, path: str, indexes: np.ndarray,
        channel_first: bool = True, backend: Optional[VisualLib] = None
    ) -> VideoData:
        """
        Get a video frames with indexes. The audio will be ignored.
        Args:
            path (``str``): Path to the video file.
            indexes (``np.ndarray``): Indexes of the video.
            channel_first (``bool``, optional): Get image dimension (T, H, W, C) if False or (T, C, H, W) if True.
                Default: True.
            backend (:class:`~tensorneko.backend.visual_lib.VisualLib`, optional): VisualLib backend.
                Default: opencv if installed, then torchvision if installed, then ffmpeg if available.
                Note: The ffmpeg backend cannot precisely index the frame. Recommend to use pytorch backend.
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
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                else:
                    if i in indexes:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if i == indexes[-1]:
                    # early stop
                    break
            assert len(frames) == len(indexes)
            cap.release()
            video = np.stack(frames, axis=0)
            if channel_first:
                video = rearrange(video, 'T H W C -> T C H W')
            return VideoData(video, None, {"video_fps": fps})
        elif backend == VisualLib.PYTORCH:
            if not VisualLib.pytorch_available():
                raise ValueError("Torchvision is not installed.")
            import torchvision
            reader = torchvision.io.VideoReader(path)
            meta = reader.get_metadata()["video"]
            fps = meta["fps"]
            first_index = indexes[0]
            first_pts = first_index / fps
            last_pts = indexes[-1] / fps
            indexes = indexes - first_index
            video = []
            for frame in takewhile(lambda x: x["pts"] <= last_pts, reader.seek(first_pts)):
                video.append(frame["data"])
            video = np.stack(video)
            video = video[indexes]  # (T C H W)

            if not channel_first:
                video = rearrange(video, 'T C H W -> T H W C')
            return VideoData(video, None, {"video_fps": fps})
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

            out = ffmpeg.input(path) \
                .filter('select', "+".join(f"eq(n,{f})" for f in indexes)) \
                .output('pipe:', vframes=len(indexes), format='rawvideo', pix_fmt='rgb24') \
                .run(quiet=True)[0]
            video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
            if channel_first:
                video = rearrange(video, 'T H W C -> T C H W')
            return VideoData(video, None, {"video_fps": fps})
        else:
            raise ValueError("Unknown backend: {}".format(backend))

    @classmethod
    @dispatch
    def with_range(cls, path: str, start: int, end: int, step: int, channel_first: bool = True,
        backend: Optional[VisualLib] = None
    ) -> VideoData:
        backend = backend or _default_video_io_backend()
        if backend == VisualLib.PYTORCH:
            if not VisualLib.pytorch_available():
                raise ValueError("Torchvision is not installed.")
            import torchvision
            reader = torchvision.io.VideoReader(path)
            meta = reader.get_metadata()["video"]
            fps = meta["fps"]
            first_pts = start / fps

            video = []
            for frame in islice(reader.seek(first_pts), start, end, step):
                video.append(frame["data"])
            video = np.stack(video)  # (T C H W)

            if not channel_first:
                video = rearrange(video, 'T C H W -> T H W C')
            return VideoData(video, None, {"video_fps": fps})
        else:
            cls.with_indexes(path, np.arange(start, end, step), channel_first, backend)

    @classmethod
    @dispatch
    def with_range(cls, path: str, end: int, channel_first: bool = True,
        backend: Optional[VisualLib] = None
    ) -> VideoData:
        return cls.with_range(path, 0, end, 1, channel_first, backend)

    def __new__(cls, path: str, channel_first: bool = False, backend: Optional[VisualLib] = None) -> VideoData:
        """Alias of :meth:`~.VideoReader.of`"""
        return cls.of(path, channel_first, backend)
