from typing import Union

import torch
from numpy import ndarray
from torch import Tensor


def iou_1d(pred: Union[Tensor, ndarray], real: Union[Tensor, ndarray]) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        pred (:class:`~torch.Tensor` | :class:``): The predicted array with [N, 2]. First column is begin, second column
            is end.
        real (:class:`~torch.Tensor` | :class:``): The label array with [L, 2]. First column is begin, second column
            is end.

    Returns:
        (:class:`~torch.Tensor` | :class:``): The iou result with [N, L].
    """
    if type(pred) is ndarray:
        pred = torch.tensor(pred)

    if type(real) is ndarray:
        real = torch.tensor(real)

    pred_begin = pred[:, 0].unsqueeze(0).T
    pred_end = pred[:, 1].unsqueeze(0).T
    real_begin = real[:, 0]
    real_end = real[:, 1]

    inner_begin = torch.maximum(pred_begin, real_begin)
    inner_end = torch.minimum(pred_end, real_end)
    outer_begin = torch.minimum(pred_begin, real_begin)
    outer_end = torch.maximum(pred_end, real_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union
