import wave
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch


def save_wav_stub(
    path: Union[str, Path],
    waveform: Any,
    sample_rate: int,
    channels_first: bool = True,
    *args,
    **kwargs,
) -> None:
    array = (
        waveform.detach().cpu().numpy()
        if isinstance(waveform, torch.Tensor)
        else np.asarray(waveform)
    )
    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 1:
        array = array[np.newaxis, :]
    if not channels_first:
        array = array.T

    if array.ndim != 2:
        raise ValueError("waveform must have 2 dimensions")

    channels, frames = array.shape
    pcm = np.clip(array, -1.0, 1.0)
    pcm = np.rint(pcm.T * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(int(channels))
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def load_wav_stub(
    path: Union[str, Path],
    channels_first: bool = True,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    audio = audio.reshape(frames, channels) / 32767.0
    audio = audio.T if channels_first else audio
    return torch.from_numpy(audio.copy()), int(sample_rate)
