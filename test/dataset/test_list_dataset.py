import unittest

from tensorneko.dataset import ListDataset


class ListDatasetTest(unittest.TestCase):

    def test_float_list(self):
        dataset = ListDataset([1.0, 2.0, 3.0])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], 1.0)
        self.assertEqual(dataset[1], 2.0)
        self.assertEqual(dataset[2], 3.0)

    def test_int_list(self):
        dataset = ListDataset([1, 2, 3])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], 1)
        self.assertEqual(dataset[1], 2)
        self.assertEqual(dataset[2], 3)

    def test_str_list(self):
        dataset = ListDataset(["1", "2", "3"])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], "1")
        self.assertEqual(dataset[1], "2")
        self.assertEqual(dataset[2], "3")

    def test_bool_list(self):
        dataset = ListDataset([True, False, True])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], True)
        self.assertEqual(dataset[1], False)
        self.assertEqual(dataset[2], True)

    def test_mixed_list(self):
        dataset = ListDataset([1, "2", 3.0])
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0], 1)
        self.assertEqual(dataset[1], "2")
        self.assertEqual(dataset[2], 3.0)

    def test_empty_list(self):
        dataset = ListDataset([])
        self.assertEqual(len(dataset), 0)

    def test_list_of_list(self):
        dataset = ListDataset([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], [1, 2, 3])
        self.assertEqual(dataset[1], [4, 5, 6])

    def test_list_of_dict(self):
        dataset = ListDataset([{"a": 1}, {"a": 2}])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], {"a": 1})
        self.assertEqual(dataset[1], {"a": 2})

    def test_list_of_tuple(self):
        dataset = ListDataset([(1, 2), (3, 4)])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], (1, 2))
        self.assertEqual(dataset[1], (3, 4))
