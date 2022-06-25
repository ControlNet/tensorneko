import subprocess

ffmpeg_available = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  shell=True).returncode == 0

if ffmpeg_available:
    from .ffmpeg import ffmpeg_command, video2frames, merge_video_audio
