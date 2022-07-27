import subprocess

ffmpeg_available = subprocess.run('ffmpeg -version', stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL, shell=True).returncode == 0

__all__ = ['ffmpeg_available']

if ffmpeg_available:
    from .ffmpeg import ffmpeg_command, video2frames, merge_video_audio, resample_video_fps, mp32wav

    __all__ += ['ffmpeg_command', 'video2frames', 'merge_video_audio', 'resample_video_fps', 'mp32wav']
