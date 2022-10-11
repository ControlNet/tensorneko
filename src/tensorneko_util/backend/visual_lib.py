from enum import Enum
from typing import Optional


class VisualLib(Enum):
    OPENCV = 1
    PYTORCH = 2
    MATPLOTLIB = 3
    PIL = 4
    FFMPEG = 5
    SKIMAGE = 6

    @staticmethod
    def opencv_available() -> bool:
        if _VisualLibAvailability._is_opencv_available is None:
            try:
                import cv2
                _VisualLibAvailability._is_opencv_available = True
            except ImportError:
                _VisualLibAvailability._is_opencv_available = False
        return _VisualLibAvailability._is_opencv_available

    @staticmethod
    def pytorch_available() -> bool:
        if _VisualLibAvailability._is_torchvision_available is None:
            try:
                import torchvision
                _VisualLibAvailability._is_torchvision_available = True
            except ImportError:
                _VisualLibAvailability._is_torchvision_available = False
        return _VisualLibAvailability._is_torchvision_available

    @staticmethod
    def matplotlib_available() -> bool:
        if _VisualLibAvailability._is_matplotlib_available is None:
            try:
                import matplotlib
                _VisualLibAvailability._is_matplotlib_available = True
            except ImportError:
                _VisualLibAvailability._is_matplotlib_available = False
        return _VisualLibAvailability._is_matplotlib_available

    @staticmethod
    def pil_available() -> bool:
        if _VisualLibAvailability._is_pil_available is None:
            try:
                import PIL
                _VisualLibAvailability._is_pil_available = True
            except ImportError:
                _VisualLibAvailability._is_pil_available = False
        return _VisualLibAvailability._is_pil_available

    @staticmethod
    def ffmpeg_available() -> bool:
        if _VisualLibAvailability._is_ffmpeg_available is None:
            from ..preprocess._ffmpeg_check import ffmpeg_available
            _VisualLibAvailability._is_ffmpeg_available = ffmpeg_available
        return _VisualLibAvailability._is_ffmpeg_available

    @staticmethod
    def skimage_available() -> bool:
        if _VisualLibAvailability._is_skimage_available is None:
            try:
                import skimage
                _VisualLibAvailability._is_skimage_available = True
            except ImportError:
                _VisualLibAvailability._is_skimage_available = False
        return _VisualLibAvailability._is_skimage_available


class _VisualLibAvailability:
    _is_opencv_available: Optional[bool] = None
    _is_torchvision_available: Optional[bool] = None
    _is_matplotlib_available: Optional[bool] = None
    _is_pil_available: Optional[bool] = None
    _is_ffmpeg_available: bool = None
    _is_skimage_available: Optional[bool] = None
