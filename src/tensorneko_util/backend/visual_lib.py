from enum import Enum
from typing import Optional


class VisualLib(Enum):
    OPENCV = 1
    PYTORCH = 2
    MATPLOTLIB = 3
    PIL = 4
    FFMPEG = 5
    SKIMAGE = 6

    _is_opencv_available: Optional[bool] = None
    _is_torchvision_available: Optional[bool] = None
    _is_matplotlib_available: bool = True  # already included in the requirements.txt
    _is_pil_available: Optional[bool] = None
    _is_ffmpeg_available: bool = None
    _is_skimage_available: Optional[bool] = None

    @classmethod
    def opencv_available(cls) -> bool:
        if cls._is_opencv_available is None:
            try:
                import cv2
                cls._is_opencv_available = True
            except ImportError:
                cls._is_opencv_available = False
        return cls._is_opencv_available

    @classmethod
    def pytorch_available(cls) -> bool:
        if cls._is_torchvision_available is None:
            try:
                import torchvision
                cls._is_torchvision_available = True
            except ImportError:
                cls._is_torchvision_available = False
        return cls._is_torchvision_available

    @classmethod
    def matplotlib_available(cls) -> bool:
        return cls._is_matplotlib_available

    @classmethod
    def pil_available(cls) -> bool:
        if cls._is_pil_available is None:
            try:
                import PIL
                cls._is_pil_available = True
            except ImportError:
                cls._is_pil_available = False
        return cls._is_pil_available

    @classmethod
    def ffmpeg_available(cls) -> bool:
        if cls._is_ffmpeg_available is None:
            from ..preprocess._ffmpeg_check import ffmpeg_available
            cls._is_ffmpeg_available = ffmpeg_available
        return cls._is_ffmpeg_available

    @classmethod
    def skimage_available(cls) -> bool:
        if cls._is_skimage_available is None:
            try:
                import skimage
                cls._is_skimage_available = True
            except ImportError:
                cls._is_skimage_available = False
        return cls._is_skimage_available
