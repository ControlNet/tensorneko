from typing import Union


class AverageMeter:
    """
    Computes and stores the average and current value.

    Attributes:
        val (float): The current value.
        avg (float): The average value.
        sum (float): The sum of all values.
        count (int): The number of values.
    """

    def __init__(self):
        self.count: int = 0
        self.sum: float = 0
        self.avg: float = 0
        self.val: float = 0

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
