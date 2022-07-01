from ..backend.visual_lib import VisualLib


def _default_image_io_backend():
    if VisualLib.opencv_available():
        backend = VisualLib.OPENCV
    elif VisualLib.matplotlib_available():
        backend = VisualLib.MATPLOTLIB
    else:
        raise ValueError("No image reader backend available.")
    return backend


def _default_video_io_backend():
    if VisualLib.opencv_available():
        return VisualLib.OPENCV
    elif VisualLib.pytorch_available():
        return VisualLib.PYTORCH
    elif VisualLib.ffmpeg_available():
        return VisualLib.FFMPEG
    else:
        raise ValueError("No backend available. Please install OpenCV, Torchvision or FFMPEG.")


def _default_audio_io_backend():
    if VisualLib.pytorch_available():
        return VisualLib.PYTORCH
    else:
        raise ValueError("No backend available. Please install Torchaudio.")
