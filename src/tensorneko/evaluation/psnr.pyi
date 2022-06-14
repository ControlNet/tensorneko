from typing import overload

from torch import Tensor


@overload
def psnr_image(pred: str, real: str) -> float: ...


@overload
def psnr_image(pred: Tensor, real: Tensor) -> float: ...


@overload
def psnr_video(pred: str, real: str, use_ffmpeg: bool = False) -> float: ...


@overload
def psnr_video(pred: Tensor, real: Tensor) -> float: ...
