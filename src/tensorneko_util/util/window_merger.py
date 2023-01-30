from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Any, List, Union, Generic, Callable
from .type import T


@dataclass
class WindowValue(Generic[T]):
    start: float
    end: float
    value: T


@dataclass
class WindowValueCache(Generic[T]):
    start: float
    end: float
    value: List[T]

    def merge_value(self, merge_func: Callable[[List[T]], T]) -> WindowValue[T]:
        return WindowValue(self.start, self.end, merge_func(self.value))

    def __eq__(self, other: WindowValueCache[T]):
        return self.start == other.start and self.end == other.end


class WindowMerger:

    def __init__(self):
        self._data: List[WindowValueCache] = []

    @classmethod
    def _get_overlap(cls, prev_start: float, prev_end: float, new_start: float, new_end: float
    ) -> Tuple[int, Tuple[Optional[float], Optional[float]]]:
        """
        Get overlap of two windows. And consider 9 cases:
        1. prev_start < new_start < prev_end < new_end
            prev: |----------|
            new :     |----------|
        2. new_start < prev_start < new_end < prev_end
            prev:     |----------|
            new : |----------|
        3. prev_start < new_start < new_end < prev_end
            prev: |--------------|
            new :     |------|
        4. new_start < prev_start < prev_end < new_end
            prev:     |------|
            new : |--------------|
        5. new_start == prev_start and new_end == prev_end
            prev: |----------|
            new : |----------|
        6. new_start == prev_start and new_end < prev_end
            prev: |----------|
            new : |------|
        7. new_start == prev_start and new_end > prev_end
            prev: |------|
            new : |----------|
        8. new_start > prev_start and new_end == prev_end
            prev: |----------|
            new :     |------|
        9. new_start < prev_start and new_end == prev_end
            prev:     |------|
            new : |----------|

        Args:
            prev_start: start of previous window
            prev_end: end of previous window
            new_start: start of new window
            new_end: end of new window

        Returns:
            (overlap_case, (overlap_start, overlap_end)):
                overlap_case is 1, 2, 3 or 4.
                overlap_start and overlap_end is the start and end of overlap window.
        """
        if prev_start < new_start < prev_end < new_end:
            return 1, (new_start, prev_end)
        elif new_start < prev_start < new_end < prev_end:
            return 2, (prev_start, new_end)
        elif prev_start < new_start < new_end < prev_end:
            return 3, (new_start, new_end)
        elif new_start < prev_start < prev_end < new_end:
            return 4, (prev_start, prev_end)
        elif new_start == prev_start and new_end == prev_end:
            return 5, (new_start, new_end)
        elif new_start == prev_start and new_end < prev_end:
            return 6, (new_start, new_end)
        elif new_start == prev_start and new_end > prev_end:
            return 7, (new_start, prev_end)
        elif new_start > prev_start and new_end == prev_end:
            return 8, (new_start, new_end)
        elif new_start < prev_start and new_end == prev_end:
            return 9, (prev_start, new_end)
        else:
            return 0, (None, None)

    @classmethod
    def _merge_step(cls, old_item: WindowValueCache, new_item: WindowValueCache
    ) -> Tuple[List[WindowValueCache], List[WindowValueCache], int]:
        start, end, value = new_item.start, new_item.end, new_item.value
        merged_items = []
        new_items = []

        overlap_case, overlap = cls._get_overlap(old_item.start, old_item.end, start, end)
        if overlap_case == 0:
            # No overlap
            merged_items.append(old_item)
            new_items.append(new_item)
        elif overlap_case == 1:
            # prev: |----------|
            # new :     |++++++++++|
            #       |---|------|+++|
            merged_items.append(WindowValueCache(old_item.start, overlap[0], old_item.value))
            merged_items.append(WindowValueCache(overlap[0], overlap[1], old_item.value + value))
            new_items.append(WindowValueCache(overlap[1], end, value))
        elif overlap_case == 2:
            # prev:     |----------|
            # new : |++++++++++|
            #       |+++|------|---|
            new_items.append(WindowValueCache(start, overlap[0], value))
            merged_items.append(WindowValueCache(overlap[0], overlap[1], old_item.value + value))
            merged_items.append(WindowValueCache(overlap[1], old_item.end, old_item.value))
        elif overlap_case == 3:
            # prev: |--------------|
            # new :     |++++++|
            #       |---|------|---|
            merged_items.append(WindowValueCache(old_item.start, overlap[0], old_item.value))
            merged_items.append(WindowValueCache(overlap[0], overlap[1], old_item.value + value))
            merged_items.append(WindowValueCache(overlap[1], old_item.end, old_item.value))
        elif overlap_case == 4:
            # prev:     |------|
            # new : |++++++++++++++|
            #       |+++|------|+++|
            new_items.append(WindowValueCache(start, overlap[0], value))
            merged_items.append(WindowValueCache(overlap[0], overlap[1], old_item.value + value))
            new_items.append(WindowValueCache(overlap[1], end, value))
        elif overlap_case == 5:
            # prev: |--------------|
            # new : |++++++++++++++|
            #       |--------------|
            merged_items.append(WindowValueCache(old_item.start, old_item.end, old_item.value + value))
        elif overlap_case == 6:
            # prev: |--------------|
            # new : |++++++|
            #       |------|-------|
            merged_items.append(WindowValueCache(old_item.start, overlap[1], old_item.value + value))
            merged_items.append(WindowValueCache(overlap[1], old_item.end, old_item.value))
        elif overlap_case == 7:
            # prev: |------|
            # new : |++++++++++++++|
            #       |------|+++++++|
            merged_items.append(WindowValueCache(old_item.start, old_item.end, old_item.value + value))
            new_items.append(WindowValueCache(overlap[1], end, value))
        elif overlap_case == 8:
            # prev: |--------------|
            # new :       |++++++++|
            #       |-----|--------|
            merged_items.append(WindowValueCache(old_item.start, overlap[0], old_item.value))
            merged_items.append(WindowValueCache(overlap[0], old_item.end, old_item.value + value))
        elif overlap_case == 9:
            # prev:         |-----|
            # new : |+++++++++++++|
            #       |+++++++|-----|
            new_items.append(WindowValueCache(start, overlap[0], value))
            merged_items.append(WindowValueCache(overlap[0], old_item.end, old_item.value + value))

        return merged_items, new_items, overlap_case

    def add(self, start: float, end: float, value: Any) -> None:
        if end <= start:
            raise ValueError("end must be larger than start")

        if len(self._data) == 0:
            self._data.append(WindowValueCache(start, end, [value]))
            return

        old_data = self._data.copy()
        old_data_overlapped = []

        new_data = []
        # find all overlap windows
        for old_item in old_data:
            overlap_case, _ = self._get_overlap(old_item.start, old_item.end, start, end)
            if overlap_case == 0:
                # if it is not overlapped, keep the old item
                new_data.append(old_item)
            else:
                # if it is overlapped, record the overlapped old items
                old_data_overlapped.append(old_item)

        # merge the overlapped old items
        new_items = [WindowValueCache(start, end, [value])]
        while len(old_data_overlapped) > 0:
            old_item = old_data_overlapped.pop(0)
            new_items_step = []

            old_data_is_overlap = False
            for new_item in new_items:
                merged_items, _new_items, overlap_case = self._merge_step(old_item, new_item)
                if overlap_case != 0:
                    old_data_is_overlap = True

                if overlap_case in (4, 5, 7, 9):
                    # if the old item is wrapped in the new item, append the old item to the result
                    new_data.extend(merged_items)
                else:
                    # if the old item is not wrapped in the new item, only append if it's not overlapped
                    new_data.extend([item for item in merged_items if item != old_item])
                new_items_step.extend(_new_items)

            if not old_data_is_overlap:
                new_data.append(old_item)

            new_items = new_items_step
            if len(new_items) == 0:
                break
        else:
            new_data.extend(new_items)

        self._data = sorted(new_data, key=lambda x: x.start)

    def merge(self) -> List[WindowValue]:
        return [each.merge_value(self.number_merge) for each in self._data]

    @staticmethod
    def number_merge(values: List[Union[int, float]]) -> float:
        return sum(values) / len(values)
