import unittest
from tensorneko_util.util.window_merger import (
    WindowMerger,
    WindowValue,
    WindowValueCache,
)


class TestWindowValue(unittest.TestCase):
    def test_window_value_creation(self):
        wv = WindowValue(start=0.0, end=1.0, value=42)
        self.assertAlmostEqual(wv.start, 0.0)
        self.assertAlmostEqual(wv.end, 1.0)
        self.assertEqual(wv.value, 42)

    def test_window_value_generic_type(self):
        wv = WindowValue(start=0.0, end=5.0, value="hello")
        self.assertEqual(wv.value, "hello")


class TestWindowValueCache(unittest.TestCase):
    def test_cache_creation(self):
        wvc = WindowValueCache(start=0.0, end=1.0, value=[1, 2, 3])
        self.assertAlmostEqual(wvc.start, 0.0)
        self.assertAlmostEqual(wvc.end, 1.0)
        self.assertEqual(wvc.value, [1, 2, 3])

    def test_merge_value(self):
        wvc = WindowValueCache(start=0.0, end=1.0, value=[2, 4, 6])
        result = wvc.merge_value(lambda vs: sum(vs) / len(vs))
        self.assertIsInstance(result, WindowValue)
        self.assertAlmostEqual(result.start, 0.0)
        self.assertAlmostEqual(result.end, 1.0)
        self.assertAlmostEqual(result.value, 4.0)

    def test_merge_value_custom_func(self):
        wvc = WindowValueCache(start=1.0, end=3.0, value=[10, 20, 30])
        result = wvc.merge_value(sum)
        self.assertEqual(result.value, 60)

    def test_equality(self):
        a = WindowValueCache(start=0.0, end=1.0, value=[1])
        b = WindowValueCache(start=0.0, end=1.0, value=[99])
        c = WindowValueCache(start=0.0, end=2.0, value=[1])
        self.assertEqual(a, b)  # same start/end, different values
        self.assertNotEqual(a, c)


class TestWindowMergerConstruction(unittest.TestCase):
    def test_default_number_merge(self):
        wm = WindowMerger()
        result = wm.number_merge([2, 4, 6])
        self.assertAlmostEqual(result, 4.0)

    def test_empty_merge(self):
        wm = WindowMerger()
        result = wm.merge()
        self.assertEqual(result, [])


class TestWindowMergerValidation(unittest.TestCase):
    def test_add_end_equals_start_raises(self):
        wm = WindowMerger()
        with self.assertRaises(ValueError):
            wm.add(5.0, 5.0, 1)

    def test_add_end_less_than_start_raises(self):
        wm = WindowMerger()
        with self.assertRaises(ValueError):
            wm.add(5.0, 3.0, 1)


class TestWindowMergerAdd(unittest.TestCase):
    def test_add_single_window(self):
        wm = WindowMerger()
        wm.add(0.0, 10.0, 1.0)
        result = wm.merge()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 10.0)
        self.assertAlmostEqual(result[0].value, 1.0)

    def test_add_non_overlapping_windows(self):
        wm = WindowMerger()
        wm.add(0.0, 5.0, 1.0)
        wm.add(10.0, 15.0, 2.0)
        result = wm.merge()
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[0].value, 1.0)
        self.assertAlmostEqual(result[1].start, 10.0)
        self.assertAlmostEqual(result[1].end, 15.0)
        self.assertAlmostEqual(result[1].value, 2.0)

    def test_add_new_before_existing(self):
        """New window entirely before existing (overlap case 0)."""
        wm = WindowMerger()
        wm.add(10.0, 20.0, 2.0)
        wm.add(0.0, 5.0, 1.0)
        result = wm.merge()
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[1].start, 10.0)
        self.assertAlmostEqual(result[1].end, 20.0)

    def test_add_new_after_existing(self):
        """New window entirely after existing (overlap case 0)."""
        wm = WindowMerger()
        wm.add(0.0, 5.0, 1.0)
        wm.add(10.0, 20.0, 2.0)
        result = wm.merge()
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[1].start, 10.0)


class TestWindowMergerOverlapCases(unittest.TestCase):
    """Test all 9 overlap cases from _get_overlap."""

    def test_case1_new_overlaps_right_of_prev(self):
        """
        Case 1: prev_start < new_start < prev_end < new_end
        prev: |----------|
        new :     |----------|
        """
        wm = WindowMerger()
        wm.add(0.0, 10.0, 2.0)
        wm.add(5.0, 15.0, 4.0)
        result = wm.merge()
        # Expected: [0,5)=2.0, [5,10)=avg(2,4)=3.0, [10,15)=4.0
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[0].value, 2.0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[1].value, 3.0)
        self.assertAlmostEqual(result[2].start, 10.0)
        self.assertAlmostEqual(result[2].end, 15.0)
        self.assertAlmostEqual(result[2].value, 4.0)

    def test_case2_new_overlaps_left_of_prev(self):
        """
        Case 2: new_start < prev_start < new_end < prev_end
        prev:     |----------|
        new : |----------|
        """
        wm = WindowMerger()
        wm.add(5.0, 15.0, 4.0)
        wm.add(0.0, 10.0, 2.0)
        result = wm.merge()
        # Expected: [0,5)=2.0, [5,10)=avg(4,2)=3.0, [10,15)=4.0
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[0].value, 2.0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[1].value, 3.0)
        self.assertAlmostEqual(result[2].start, 10.0)
        self.assertAlmostEqual(result[2].end, 15.0)
        self.assertAlmostEqual(result[2].value, 4.0)

    def test_case3_new_contained_within_prev(self):
        """
        Case 3: prev_start < new_start < new_end < prev_end
        prev: |--------------|
        new :     |------|
        """
        wm = WindowMerger()
        wm.add(0.0, 20.0, 6.0)
        wm.add(5.0, 15.0, 4.0)
        result = wm.merge()
        # Expected: [0,5)=6.0, [5,15)=avg(6,4)=5.0, [15,20)=6.0
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[0].value, 6.0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[1].end, 15.0)
        self.assertAlmostEqual(result[1].value, 5.0)
        self.assertAlmostEqual(result[2].start, 15.0)
        self.assertAlmostEqual(result[2].end, 20.0)
        self.assertAlmostEqual(result[2].value, 6.0)

    def test_case4_new_contains_prev(self):
        """
        Case 4: new_start < prev_start < prev_end < new_end
        prev:     |------|
        new : |--------------|
        """
        wm = WindowMerger()
        wm.add(5.0, 15.0, 4.0)
        wm.add(0.0, 20.0, 6.0)
        result = wm.merge()
        # Expected: [0,5)=6.0, [5,15)=avg(4,6)=5.0, [15,20)=6.0
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[0].value, 6.0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[1].end, 15.0)
        self.assertAlmostEqual(result[1].value, 5.0)
        self.assertAlmostEqual(result[2].start, 15.0)
        self.assertAlmostEqual(result[2].end, 20.0)
        self.assertAlmostEqual(result[2].value, 6.0)

    def test_case5_exact_match(self):
        """
        Case 5: new_start == prev_start and new_end == prev_end
        prev: |----------|
        new : |----------|
        """
        wm = WindowMerger()
        wm.add(0.0, 10.0, 2.0)
        wm.add(0.0, 10.0, 8.0)
        result = wm.merge()
        # Expected: [0,10)=avg(2,8)=5.0
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 10.0)
        self.assertAlmostEqual(result[0].value, 5.0)

    def test_case6_same_start_new_shorter(self):
        """
        Case 6: new_start == prev_start and new_end < prev_end
        prev: |----------|
        new : |------|
        """
        wm = WindowMerger()
        wm.add(0.0, 10.0, 4.0)
        wm.add(0.0, 6.0, 2.0)
        result = wm.merge()
        # Expected: [0,6)=avg(4,2)=3.0, [6,10)=4.0
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 6.0)
        self.assertAlmostEqual(result[0].value, 3.0)
        self.assertAlmostEqual(result[1].start, 6.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[1].value, 4.0)

    def test_case7_same_start_new_longer(self):
        """
        Case 7: new_start == prev_start and new_end > prev_end
        prev: |------|
        new : |----------|
        """
        wm = WindowMerger()
        wm.add(0.0, 6.0, 4.0)
        wm.add(0.0, 10.0, 2.0)
        result = wm.merge()
        # Expected: [0,6)=avg(4,2)=3.0, [6,10)=2.0
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 6.0)
        self.assertAlmostEqual(result[0].value, 3.0)
        self.assertAlmostEqual(result[1].start, 6.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[1].value, 2.0)

    def test_case8_same_end_new_starts_later(self):
        """
        Case 8: new_start > prev_start and new_end == prev_end
        prev: |----------|
        new :     |------|
        """
        wm = WindowMerger()
        wm.add(0.0, 10.0, 4.0)
        wm.add(4.0, 10.0, 2.0)
        result = wm.merge()
        # Expected: [0,4)=4.0, [4,10)=avg(4,2)=3.0
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 4.0)
        self.assertAlmostEqual(result[0].value, 4.0)
        self.assertAlmostEqual(result[1].start, 4.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[1].value, 3.0)

    def test_case9_same_end_new_starts_earlier(self):
        """
        Case 9: new_start < prev_start and new_end == prev_end
        prev:     |------|
        new : |----------|
        """
        wm = WindowMerger()
        wm.add(4.0, 10.0, 4.0)
        wm.add(0.0, 10.0, 2.0)
        result = wm.merge()
        # Expected: [0,4)=2.0, [4,10)=avg(4,2)=3.0
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 4.0)
        self.assertAlmostEqual(result[0].value, 2.0)
        self.assertAlmostEqual(result[1].start, 4.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[1].value, 3.0)


class TestWindowMergerMultipleWindows(unittest.TestCase):
    def test_new_spans_multiple_existing(self):
        """New window overlaps multiple existing windows."""
        wm = WindowMerger()
        wm.add(0.0, 5.0, 2.0)
        wm.add(10.0, 15.0, 4.0)
        # New window spans both
        wm.add(3.0, 12.0, 6.0)
        result = wm.merge()
        # [0,3)=2.0, [3,5)=avg(2,6)=4.0, [5,10)=6.0, [10,12)=avg(4,6)=5.0, [12,15)=4.0
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 3.0)
        self.assertAlmostEqual(result[0].value, 2.0)
        self.assertAlmostEqual(result[1].start, 3.0)
        self.assertAlmostEqual(result[1].end, 5.0)
        self.assertAlmostEqual(result[1].value, 4.0)
        self.assertAlmostEqual(result[2].start, 5.0)
        self.assertAlmostEqual(result[2].end, 10.0)
        self.assertAlmostEqual(result[2].value, 6.0)
        self.assertAlmostEqual(result[3].start, 10.0)
        self.assertAlmostEqual(result[3].end, 12.0)
        self.assertAlmostEqual(result[3].value, 5.0)
        self.assertAlmostEqual(result[4].start, 12.0)
        self.assertAlmostEqual(result[4].end, 15.0)
        self.assertAlmostEqual(result[4].value, 4.0)

    def test_three_sequential_overlaps(self):
        """Three windows that each overlap the previous."""
        wm = WindowMerger()
        wm.add(0.0, 10.0, 3.0)
        wm.add(5.0, 15.0, 3.0)
        wm.add(10.0, 20.0, 3.0)
        result = wm.merge()
        # [0,5)=3.0, [5,10)=avg(3,3)=3.0, [10,15)=avg(3,3)=3.0, [15,20)=3.0
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0].start, 0.0)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[1].end, 10.0)
        self.assertAlmostEqual(result[2].start, 10.0)
        self.assertAlmostEqual(result[2].end, 15.0)
        self.assertAlmostEqual(result[3].start, 15.0)
        self.assertAlmostEqual(result[3].end, 20.0)

    def test_adjacent_windows_no_gap(self):
        """Adjacent windows (end == start of next) — no overlap (case 0)."""
        wm = WindowMerger()
        wm.add(0.0, 5.0, 1.0)
        wm.add(5.0, 10.0, 2.0)
        result = wm.merge()
        # Adjacent, no overlap
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].end, 5.0)
        self.assertAlmostEqual(result[1].start, 5.0)
        self.assertAlmostEqual(result[0].value, 1.0)
        self.assertAlmostEqual(result[1].value, 2.0)

    def test_sorted_output(self):
        """Result should be sorted by start time."""
        wm = WindowMerger()
        wm.add(20.0, 30.0, 3.0)
        wm.add(0.0, 10.0, 1.0)
        wm.add(10.0, 20.0, 2.0)
        result = wm.merge()
        for i in range(len(result) - 1):
            self.assertLessEqual(result[i].start, result[i + 1].start)


class TestWindowMergerMerge(unittest.TestCase):
    def test_merge_default_averaging(self):
        """Default merge function averages overlapping values."""
        wm = WindowMerger()
        wm.add(0.0, 10.0, 10.0)
        wm.add(0.0, 10.0, 20.0)
        result = wm.merge()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].value, 15.0)

    def test_merge_three_overlapping(self):
        """Three identical windows merged → average of 3 values."""
        wm = WindowMerger()
        wm.add(0.0, 10.0, 3.0)
        wm.add(0.0, 10.0, 6.0)
        wm.add(0.0, 10.0, 9.0)
        result = wm.merge()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].value, 6.0)

    def test_custom_merge_function(self):
        """Override number_merge with custom sum function."""
        wm = WindowMerger()
        wm.number_merge = sum
        wm.add(0.0, 10.0, 2.0)
        wm.add(0.0, 10.0, 3.0)
        result = wm.merge()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].value, 5.0)

    def test_custom_merge_function_max(self):
        """Override number_merge with max function."""
        wm = WindowMerger()
        wm.number_merge = max
        wm.add(0.0, 10.0, 2.0)
        wm.add(0.0, 10.0, 8.0)
        result = wm.merge()
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].value, 8.0)


class TestWindowMergerMergeStepCase0(unittest.TestCase):
    """Test _merge_step case 0 (no overlap) via add with fragmented new_items."""

    def test_new_spans_three_with_gaps(self):
        """When new window spans 3 old windows with gaps, _merge_step sees case 0
        for fragments that don't overlap with a particular old_item.
        Covers window_merger.py lines 108-109."""
        wm = WindowMerger()
        wm.add(0.0, 3.0, 1.0)
        wm.add(7.0, 10.0, 2.0)
        wm.add(20.0, 25.0, 3.0)
        # New window spans all three with gaps
        wm.add(0.0, 25.0, 4.0)
        result = wm.merge()
        # Should have segments: [0,3], [3,7], [7,10], [10,20], [20,25]
        # with merged values where overlapping
        self.assertTrue(len(result) >= 5)
        # Verify the non-overlapping gap segments exist
        starts = [r.start for r in result]
        ends = [r.end for r in result]
        self.assertIn(0.0, starts)
        self.assertIn(25.0, ends)


class TestGetOverlapDirect(unittest.TestCase):
    """Directly test _get_overlap for completeness."""

    def test_no_overlap(self):
        case, bounds = WindowMerger._get_overlap(0, 5, 10, 15)
        self.assertEqual(case, 0)
        self.assertIsNone(bounds[0])
        self.assertIsNone(bounds[1])

    def test_all_nine_cases(self):
        # Case 1: prev_start < new_start < prev_end < new_end
        self.assertEqual(WindowMerger._get_overlap(0, 10, 5, 15)[0], 1)
        # Case 2: new_start < prev_start < new_end < prev_end
        self.assertEqual(WindowMerger._get_overlap(5, 15, 0, 10)[0], 2)
        # Case 3: prev_start < new_start < new_end < prev_end
        self.assertEqual(WindowMerger._get_overlap(0, 20, 5, 15)[0], 3)
        # Case 4: new_start < prev_start < prev_end < new_end
        self.assertEqual(WindowMerger._get_overlap(5, 15, 0, 20)[0], 4)
        # Case 5: exact match
        self.assertEqual(WindowMerger._get_overlap(0, 10, 0, 10)[0], 5)
        # Case 6: same start, new shorter
        self.assertEqual(WindowMerger._get_overlap(0, 10, 0, 5)[0], 6)
        # Case 7: same start, new longer
        self.assertEqual(WindowMerger._get_overlap(0, 5, 0, 10)[0], 7)
        # Case 8: same end, new starts later
        self.assertEqual(WindowMerger._get_overlap(0, 10, 5, 10)[0], 8)
        # Case 9: same end, new starts earlier
        self.assertEqual(WindowMerger._get_overlap(5, 10, 0, 10)[0], 9)

    def test_overlap_bounds_case1(self):
        case, bounds = WindowMerger._get_overlap(0, 10, 5, 15)
        self.assertAlmostEqual(bounds[0], 5.0)
        self.assertAlmostEqual(bounds[1], 10.0)

    def test_overlap_bounds_case4(self):
        case, bounds = WindowMerger._get_overlap(5, 15, 0, 20)
        self.assertAlmostEqual(bounds[0], 5.0)
        self.assertAlmostEqual(bounds[1], 15.0)


if __name__ == "__main__":
    unittest.main()
