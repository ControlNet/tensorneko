import os
import subprocess
from pathlib import Path
from subprocess import Popen
from typing import Optional, List


def ffmpeg_command(ffmpeg_args: List[str] = None, **kwargs) -> Popen:
    """
    Run ffmpeg with the given arguments.

    Args:
        ffmpeg_args (``list``, optional): Additional arguments for ffmpeg.
        **kwargs: Keyword arguments for ~:class:`subprocess.Popen`.

    Returns:
        ``Popen``: The subprocess of ffmpeg.
    """

    ffmpeg_args = ffmpeg_args or []

    return subprocess.Popen(["ffmpeg"] + ffmpeg_args, **kwargs)


def video2frames(video_path: str, output_dir: Optional[str] = None, frame_name_pattern: str = "%04d.png",
    ffmpeg_args: List[str] = None
) -> Popen:
    """
    Run ffmpeg to extract frames from a video.

    Args:
        video_path (``str``): Path to the video.
        output_dir (``str``, optional): Path to the output directory. If not specified, the output directory will be
            the name of the video. E.g. "path/to/video.mp4" -> "path/to/video".
        frame_name_pattern (``str``, optional): Pattern of the frame name used in ffmpeg command.
        ffmpeg_args (``list``, optional): Additional arguments for ffmpeg.

    Returns:
        ``Popen``: The subprocess of ffmpeg.
    """

    if output_dir is None:
        path = Path(video_path)
        output_dir = os.path.join(str(path.parent), str("".join(path.name.split(".")[:-1])))

    ffmpeg_args = ffmpeg_args or []

    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True)

    args = [
        "-i", video_path, os.path.join(output_dir, frame_name_pattern), *ffmpeg_args
    ]

    return ffmpeg_command(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def merge_video_audio(video_path: str, audio_path: str, output_path: str, shortest: bool = False,
    ffmpeg_args: List[str] = None) -> Popen:
    """
    Run ffmpeg to merge video and audio.

    Args:
        video_path (``str``): Path to the video.
        audio_path (``str``): Path to the audio.
        output_path (``str``): Path to the output video.
        shortest (``bool``, optional): If True, the output video will be the shortest possible.
        ffmpeg_args (``list``, optional): Additional arguments for ffmpeg.

    Returns:
        ``Popen``: The subprocess of ffmpeg.
    """

    ffmpeg_args = ffmpeg_args or []

    if shortest:
        ffmpeg_args.append("-shortest")

    args = [
        "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", output_path, *ffmpeg_args
    ]

    return ffmpeg_command(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def resample_video_fps(video_path: str, output_path: str, fps: float, ffmpeg_args: List[str] = None) -> Popen:
    """
    Run ffmpeg to resample video to a specified FPS.

    Args:
        video_path (``str``): Path to the video.
        output_path (``str``): Path to the output video.
        fps (``float``): FPS of the output video.
        ffmpeg_args (``list``, optional): Additional arguments for ffmpeg.

    Returns:
        ``Popen``: The subprocess of ffmpeg.
    """

    ffmpeg_args = ffmpeg_args or []

    args = [
        "-i", video_path, "-filter:v", f"fps={fps}", output_path, *ffmpeg_args
    ]

    return ffmpeg_command(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mp32wav(mp3_path: str, wav_path: str, ffmpeg_args: List[str] = None) -> Popen:
    """
    Run ffmpeg to convert mp3 to wav.

    Args:
        mp3_path (``str``): Path to the mp3.
        wav_path (``str``): Path to the wav.
        ffmpeg_args (``list``, optional): Additional arguments for ffmpeg.

    Returns:
        ``Popen``: The subprocess of ffmpeg.
    """

    ffmpeg_args = ffmpeg_args or []

    args = [
        "-i", mp3_path, wav_path, *ffmpeg_args
    ]
    return ffmpeg_command(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
