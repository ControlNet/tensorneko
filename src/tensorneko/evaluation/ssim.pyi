from typing import overload

from torch import Tensor


@overload
def ssim_image(pred: str, real: str) -> float: ...


@overload
def ssim_image(pred: Tensor, real: Tensor) -> float: ...


@overload
def ssim_video(pred: str, real: str, use_ffmpeg: bool = False) -> float: ...


@overload
def ssim_video(pred: Tensor, real: Tensor) -> float: ...
