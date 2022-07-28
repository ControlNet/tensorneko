from ._ffmpeg_check import ffmpeg_available
from .video import frames2video

__all__ = ['ffmpeg_available', 'frames2video']

if ffmpeg_available:
    from .ffmpeg import ffmpeg_command, video2frames, merge_video_audio, resample_video_fps, mp32wav

    __all__ += ['ffmpeg_command', 'video2frames', 'merge_video_audio', 'resample_video_fps', 'mp32wav']
