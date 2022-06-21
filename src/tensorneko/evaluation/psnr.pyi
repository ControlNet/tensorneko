from typing import overload

from torch import Tensor

from .enum import Reduction


@overload
def psnr_image(pred: str, real: str) -> Tensor: ...


@overload
def psnr_image(pred: Tensor, real: Tensor, reduction: Reduction = Reduction.MEAN) -> Tensor: ...


@overload
def psnr_video(pred: str, real: str, use_ffmpeg: bool = False) -> Tensor: ...


@overload
def psnr_video(pred: Tensor, real: Tensor) -> Tensor: ...
