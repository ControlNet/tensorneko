from typing import Union


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val: Union[int, float], n: int = 1) -> None:
        """
        Add new values to the average meter.

        Args:
            val(``int`` | ``float``): The new value to be added.
            n(``int``, optional): The number of values to be added.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
