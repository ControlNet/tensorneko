from ._ffmpeg_check import ffmpeg_available
from .video import frames2video
from .image import rgb2gray, rgb2gray_batch
from .crop import crop_with_padding

__all__ = ['ffmpeg_available', 'frames2video', 'rgb2gray', 'rgb2gray_batch', 'crop_with_padding']

if ffmpeg_available:
    from .ffmpeg import ffmpeg_command, video2frames, merge_video_audio, resample_video_fps, mp32wav

    __all__ += ['ffmpeg_command', 'video2frames', 'merge_video_audio', 'resample_video_fps', 'mp32wav']
