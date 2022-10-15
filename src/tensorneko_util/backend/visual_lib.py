import subprocess
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
        if _VisualLibAvailability.is_opencv_available is None:
            try:
                import cv2
                _VisualLibAvailability.is_opencv_available = True
            except ImportError:
                _VisualLibAvailability.is_opencv_available = False
        return _VisualLibAvailability.is_opencv_available

    @staticmethod
    def pytorch_available() -> bool:
        if _VisualLibAvailability.is_torchvision_available is None:
            try:
                import torchvision
                _VisualLibAvailability.is_torchvision_available = True
            except ImportError:
                _VisualLibAvailability.is_torchvision_available = False
        return _VisualLibAvailability.is_torchvision_available

    @staticmethod
    def matplotlib_available() -> bool:
        if _VisualLibAvailability.is_matplotlib_available is None:
            try:
                import matplotlib
                _VisualLibAvailability.is_matplotlib_available = True
            except ImportError:
                _VisualLibAvailability.is_matplotlib_available = False
        return _VisualLibAvailability.is_matplotlib_available

    @staticmethod
    def pil_available() -> bool:
        if _VisualLibAvailability.is_pil_available is None:
            try:
                import PIL
                _VisualLibAvailability.is_pil_available = True
            except ImportError:
                _VisualLibAvailability.is_pil_available = False
        return _VisualLibAvailability.is_pil_available

    @staticmethod
    def ffmpeg_available() -> bool:
        if _VisualLibAvailability.is_ffmpeg_available is None:
            ffmpeg_available = subprocess.run('ffmpeg -version', stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, shell=True).returncode == 0
            _VisualLibAvailability.is_ffmpeg_available = ffmpeg_available
        return _VisualLibAvailability.is_ffmpeg_available

    @staticmethod
    def skimage_available() -> bool:
        if _VisualLibAvailability.is_skimage_available is None:
            try:
                import skimage
                _VisualLibAvailability.is_skimage_available = True
            except ImportError:
                _VisualLibAvailability.is_skimage_available = False
        return _VisualLibAvailability.is_skimage_available


class _VisualLibAvailability:
    is_opencv_available: Optional[bool] = None
    is_torchvision_available: Optional[bool] = None
    is_matplotlib_available: Optional[bool] = None
    is_pil_available: Optional[bool] = None
    is_ffmpeg_available: Optional[bool] = None
    is_skimage_available: Optional[bool] = None
