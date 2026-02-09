"""Tests for json_data constructor branches — List[non-json_data] inner type."""

import unittest
from typing import List

from tensorneko_util.io.json.json_data import json_data


@json_data
class TaggedItem:
    name: str
    tags: List[str]


@json_data
class NestedPlain:
    matrix: List[List[int]]


class TestJsonDataListNonJsonData(unittest.TestCase):
    """Cover json_data.py lines 46-47, 52: List[X] where X is NOT json_data."""

    def test_list_of_str_field(self):
        """Field tags: List[str] — inner_type is str, not json_data → lines 44-47, 52."""
        item = TaggedItem({"name": "foo", "tags": ["a", "b", "c"]})
        self.assertEqual(item.name, "foo")
        self.assertEqual(item.tags, ["a", "b", "c"])

    def test_list_of_list_of_int_field(self):
        """Field matrix: List[List[int]] — inner_type is List[int], inner_inner_type is int.
        int doesn't have is_json_data → lines 37-38 (AttributeError), then v = d[k] (line 42)."""
        obj = NestedPlain({"matrix": [[1, 2], [3, 4]]})
        self.assertEqual(obj.matrix, [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
