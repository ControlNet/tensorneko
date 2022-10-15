from typing import Iterable, Optional

from ..backend.visual_lib import VisualLib
from ..io._default_backends import _default_video_io_backend


def frames2video(frame_paths: Iterable[str], output_path: str, fps: int, backend: Optional[VisualLib] = None) -> None:
    """
    Convert frames to video.

    Args:
        frame_paths (``Iterable[str]``): Paths of frames.
        output_path (``str``): Path of output video.
        fps (``int``): Frames per second.
        backend (:class:`~tensorneko_util.backend.visual_lib.VisualLib`, optional): The backend library for saving.
            Default: OPENCV if installed else PYTORCH then FFMPEG.
    """
    backend = backend or _default_video_io_backend()

    if backend == VisualLib.OPENCV:
        if not VisualLib.opencv_available():
            raise ValueError("OpenCV is not installed.")

        import cv2
        ext = output_path.split('.')[-1]
        if ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        else:
            raise ValueError("Video ext only supports 'avi' or 'mp4'.")

        video = None

        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if video is None:
                video = cv2.VideoWriter(output_path, fourcc, fps, img.shape[1::-1])
            video.write(img)

        video.release()

    elif backend == VisualLib.PYTORCH:
        if not VisualLib.pytorch_available():
            raise ValueError("Torchvision is not installed.")

        import torchvision
        import torch

        frames = []

        for frame_path in frame_paths:
            frames.append(torchvision.io.read_image(frame_path))

        torchvision.io.write_video(output_path, torch.stack(frames, dim=0).permute(0, 2, 3, 1), fps=fps)

    elif backend == VisualLib.FFMPEG:
        raise NotImplementedError("FFMPEG is not implemented yet.")

    else:
        raise ValueError("Supported backend is only OpenCV and PyTorch.")
