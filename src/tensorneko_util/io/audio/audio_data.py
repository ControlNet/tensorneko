from dataclasses import dataclass

from ...util.type import T_ARRAY


@dataclass
class AudioData:
    """
    Output audio data for :class:`~.audio_reader.AudioReader`.

    Attributes:
        audio (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): audio float tensor of (C, T)
        sample_rate (``int``): sample rate of the audio

    """
    audio: T_ARRAY
    sample_rate: int

    def __iter__(self):
        return iter((self.audio, self.sample_rate))
