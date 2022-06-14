import re
import subprocess

from torch import Tensor
from torchmetrics.functional import psnr

from tensorneko_util.preprocess import ffmpeg_available
from tensorneko_util.util import dispatch

from ..io import read
from ..preprocess import padding_video, PaddingMethod


@dispatch
def psnr_image(pred: str, real: str) -> float:
    """
    Calculate PSNR of an image.

    Args:
        pred (``str``): Path to the predicted image.
        real (``str``): Path to the real image.

    Returns:
        ``float``: The psnr of the image.
    """
    pred_image = read.image(pred).image
    real_image = read.image(real).image
    return float(psnr(pred_image, real_image, data_range=1.0))


@dispatch
def psnr_image(pred: Tensor, real: Tensor) -> float:
    """
    Calculate PSNR of an image.

    Args:
        pred (:class:`~torch.Tensor`): Predicted image tensor. (C, H, W)
        real (:class:`~torch.Tensor`): Real image tensor. (C, H, W)

    Returns:
        ``float``: The psnr of the image.
    """
    return float(psnr(pred, real, data_range=1.0))


@dispatch
def psnr_video(pred: str, real: str, use_ffmpeg: bool = False) -> float:
    """
    Calculate PSNR of a video.

    Args:
        pred (``str``): Path to the predicted video.
        real (``str``): Path to the real video.
        use_ffmpeg (``bool``, optional): Whether to use ffmpeg to calculate psnr.

    Returns:
        ``float``: The psnr of the video.
    """
    if use_ffmpeg:
        if not ffmpeg_available:
            raise RuntimeError("ffmpeg is not found.")

        p = subprocess.run(["-i", pred, "-i", real, "-filter_complex", "psnr", "-f", "null", "/dev/null"],
                           capture_output=True)
        psnr_out = p.stdout.decode().split("\n")[-2]
        return float(re.search(r" average:([\d.]+) ", psnr_out)[1])
    else:
        pred_video = read.video(pred).video  # (T, C, H, W)
        real_video = read.video(real).video
        return psnr_video(pred_video, real_video)


@dispatch
def psnr_video(pred: Tensor, real: Tensor) -> float:
    """
    Calculate PSNR of a video.

    Args:
        pred (:class:`~torch.Tensor`): Predicted video tensor. (T, C, H, W)
        real (:class:`~torch.Tensor`): Real video tensor. (T, C, H, W)

        Returns:
            ``float``: The psnr of the video.
    """
    real_video = padding_video(real, pred.shape[0], PaddingMethod.SAME)
    psnr_results = []
    for i in range(pred.shape[0]):
        psnr_results.append(psnr_image(pred[i], real_video[i]))
    return sum(psnr_results) / len(psnr_results)
