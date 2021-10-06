from typing import Tuple


def get_iou_1d(pred: Tuple[float, float], real: Tuple[float, float]) -> float:
    pred_begin, pred_end = pred
    real_begin, real_end = real

    if pred_begin <= real_begin:
        outer_begin = pred_begin
        inner_begin = real_begin
    else:
        outer_begin = real_begin
        inner_begin = pred_begin

    if pred_end <= real_end:
        outer_end = real_end
        inner_end = pred_end
    else:
        outer_end = pred_end
        inner_end = real_end

    inter = inner_end - inner_begin
    union = outer_end - outer_begin
    return inter / union

