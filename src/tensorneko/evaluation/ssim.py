import re
import subprocess

import torch
from torch import Tensor, tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim

from tensorneko_util.backend import VisualLib
from tensorneko_util.util import dispatch
from .enum import Reduction
from ..io import read
from ..preprocess import padding_video, PaddingMethod


@dispatch
def ssim_image(pred: str, real: str) -> Tensor:
    """
    Calculate SSIM of an image.

    Args:
        pred (``str``): Path to the predicted image.
        real (``str``): Path to the real image.

    Returns:
        :class:`~torch.Tensor`: The ssim of the image.
    """
    pred_image = read.image(pred).unsqueeze(0)
    real_image = read.image(real).unsqueeze(0)
    return ssim(pred_image, real_image, data_range=1.0, reduction="elementwise_mean")


@dispatch
def ssim_image(pred: Tensor, real: Tensor, reduction: Reduction = Reduction.MEAN) -> Tensor:
    """
    Calculate SSIM of an image or a batch of images.

    Args:
        pred (:class:`~torch.Tensor`): Predicted images tensor. (B, C, H, W) or (C, H, W)
        real (:class:`~torch.Tensor`): Real images tensor. (B, C, H, W) or (C, H, W)
        reduction: (:class:`tensorneko.evaluation.enum.Reduction`, optional): Reduction method.
            Default: ``Reduction.MEAN``.

    Returns:
        :class:`~torch.Tensor`: The ssim of the images.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)

    if real.dim() == 3:
        real = real.unsqueeze(0)

    assert pred.shape[0] == real.shape[0], "The number of images in pred and real must be equal."

    if pred.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        pred = pred.float() / 255.

    if real.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        real = real.float() / 255.

    reduction_method = reduction.value

    if reduction_method == "mean":
        reduction_method = "elementwise_mean"

    return ssim(pred, real, data_range=1.0, reduction=reduction_method)


@dispatch
def ssim_video(pred: str, real: str, use_ffmpeg: bool = False) -> Tensor:
    """
    Calculate SSIM of a video.

    Args:
        pred (``str``): Path to the predicted video.
        real (``str``): Path to the real video.
        use_ffmpeg (``bool``, optional): Whether to use ffmpeg to calculate ssim.

    Returns:
        :class:`~torch.Tensor`: The ssim of the video.
    """
    if use_ffmpeg:
        if not VisualLib.ffmpeg_available():
            raise RuntimeError("ffmpeg is not found.")

        p = subprocess.run(["ffmpeg", "-i", pred, "-i", real, "-filter_complex", "ssim", "-f", "null", "/dev/null"],
                           capture_output=True)
        ssim_out = p.stderr.decode().split("\n")[-2]
        return tensor(float(re.search(r" All:([\d.]+) ", ssim_out)[1]))
    else:
        pred_video = tensor(read.video(pred).video)  # (T, C, H, W)
        real_video = tensor(read.video(real).video)
        return ssim_video(pred_video, real_video)


@dispatch
def ssim_video(pred: Tensor, real: Tensor) -> Tensor:
    """
    Calculate SSIM of a video.

    Args:
        pred (:class:`~torch.Tensor`): Predicted video tensor. (T, C, H, W)
        real (:class:`~torch.Tensor`): Real video tensor. (T, C, H, W)

        Returns:
            :class:`~torch.Tensor`: The ssim of the video.
    """
    if pred.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        pred = pred.float() / 255.0
    if real.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        real = real.float() / 255.0

    real_video = padding_video(real, pred.shape[0], PaddingMethod.SAME)
    return ssim_image(pred, real_video, reduction=Reduction.MEAN)
