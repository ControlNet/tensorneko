from numpy import ndarray
from torch import Tensor

from tensorneko_util.util import dispatch, Eval

from tensorneko_util.io import read


@Eval.later
def _secs_encoder():
    from resemblyzer import VoiceEncoder
    return VoiceEncoder()


@dispatch
def secs(pred: str, real: str) -> float:
    from resemblyzer import VoiceEncoder, preprocess_wav
    pred_audio = preprocess_wav(read.audio(pred).audio[0].numpy())
    real_audio = preprocess_wav(read.audio(real).audio[0].numpy())
    return _secs_compute(pred_audio, real_audio)


@dispatch
def secs(pred: Tensor, real: Tensor) -> float:
    return secs(pred.numpy(), real.numpy())


@dispatch
def secs(pred: ndarray, real: ndarray) -> float:
    from resemblyzer import VoiceEncoder, preprocess_wav
    if len(pred.shape) == 2:
        if pred.shape[0] == 1:
            pred = pred.squeeze(0)
        elif pred.shape[1] == 1:
            pred = pred.squeeze(1)
        else:
            raise ValueError("The input audio must be mono.")

    if len(real.shape) == 2:
        if real.shape[0] == 1:
            real = real.squeeze(0)
        elif real.shape[1] == 1:
            real = real.squeeze(1)
        else:
            raise ValueError("The input audio must be mono.")

    pred_audio = preprocess_wav(pred)
    real_audio = preprocess_wav(real)

    return _secs_compute(pred_audio, real_audio)


def _secs_compute(pred_audio: ndarray, real_audio: ndarray) -> float:
    encoder = _secs_encoder.value
    real_embed = encoder.embed_utterance(real_audio)
    pred_embed = encoder.embed_utterance(pred_audio)

    return float((real_embed * pred_embed).sum())
