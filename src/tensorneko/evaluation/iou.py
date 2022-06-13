from typing import Union

import torch
from numpy import ndarray
from torch import Tensor


def iou_1d(proposal: Union[Tensor, ndarray], target: Union[Tensor, ndarray]) -> Tensor:
    """
    Calculate 1D IOU for N proposals with L labels.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The predicted array with [M, 2]. First column is
            beginning, second column is end.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The label array with [N, 2]. First column is
            beginning, second column is end.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is ndarray:
        proposal = torch.tensor(proposal)

    if type(target) is ndarray:
        target = torch.tensor(target)

    proposal_begin = proposal[:, 0].unsqueeze(0).T
    proposal_end = proposal[:, 1].unsqueeze(0).T
    target_begin = target[:, 0]
    target_end = target[:, 1]

    inner_begin = torch.maximum(proposal_begin, target_begin)
    inner_end = torch.minimum(proposal_end, target_end)
    outer_begin = torch.minimum(proposal_begin, target_begin)
    outer_end = torch.maximum(proposal_end, target_end)

    inter = torch.clamp(inner_end - inner_begin, min=0.)
    union = outer_end - outer_begin
    return inter / union


def iou_2d(proposal: Union[Tensor, ndarray], target: Union[Tensor, ndarray]) -> Tensor:
    """
    Calculate 2D IOU for M proposals with N targets.

    Args:
        proposal (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The proposals array with shape [M, 4]. The 4
            columns represents x1, y1, x2, y2.
        target (:class:`~torch.Tensor` | :class:`~numpy.ndarray`): The targets array with shape [N, 4]. The 4 columns
            represents x1, y1, x2, y2.

    Returns:
        :class:`~torch.Tensor`: The iou result with [M, N].
    """
    if type(proposal) is ndarray:
        proposal = torch.tensor(proposal)

    if type(target) is ndarray:
        target = torch.tensor(target)

    proposal_x1 = proposal[:, 0]
    proposal_y1 = proposal[:, 1]
    proposal_x2 = proposal[:, 2]
    proposal_y2 = proposal[:, 3]

    target_x1 = target[:, 0].unsqueeze(0).T
    target_y1 = target[:, 1].unsqueeze(0).T
    target_x2 = target[:, 2].unsqueeze(0).T
    target_y2 = target[:, 3].unsqueeze(0).T

    inner_x1 = torch.maximum(proposal_x1, target_x1)
    inner_y1 = torch.maximum(proposal_y1, target_y1)
    inner_x2 = torch.minimum(proposal_x2, target_y2)
    inner_y2 = torch.minimum(proposal_y2, target_y2)

    area_proposal = (proposal_x2 - proposal_x1) * (proposal_y2 - proposal_y1)
    area_target = (target_x2 - target_x1) * (target_y2 - target_y1)

    inter_x = torch.clamp(inner_x2 - inner_x1, min=0.)
    inter_y = torch.clamp(inner_y2 - inner_y1, min=0.)
    inter = inter_x * inter_y

    union = area_proposal + area_target - inter

    return inter / union
