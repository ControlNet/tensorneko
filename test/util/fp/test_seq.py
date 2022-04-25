import unittest

from tensorneko_util.backend.parallel import ParallelType
from tensorneko_util.util import Seq


class UtilSeqTest(unittest.TestCase):
    def test_seq_from_elements(self):
        s = Seq.of(1, 2, 3)
        self.assertEqual(s.to_list(), [1, 2, 3])

    def test_seq_from_iter(self):
        s = Seq([1, 2, 3])
        self.assertEqual(s.to_list(), [1, 2, 3])

    def test_seq_lshift(self):
        s = Seq.of(1, 2, 3)
        s = s << [1, 2, 3]
        self.assertEqual(s.to_list(), [1, 2, 3, 1, 2, 3])
        s = s << iter([1, 2, 3])
        self.assertEqual(s.to_list(), [1, 2, 3, 1, 2, 3, 1, 2, 3])

    def test_seq_indexing(self):
        s = Seq.of(1, 2, 3, 4, 5)
        self.assertEqual(s[1], 2)
        self.assertEqual(s[1:4].to_list(), Seq.of(2, 3, 4).to_list())

    def test_seq_print(self):
        s = Seq.of(1, 2, 3)
        self.assertEqual(str(s), "Seq(1, 2, 3)")
        self.assertEqual(s[1], 2)
        self.assertEqual(str(s), "Seq(1, 2, 3)")

    def test_seq_map(self):
        s = Seq.of(1, 2, 3)
        s = s.map(lambda x: x + 1)
        self.assertEqual(s.to_list(), [2, 3, 4])

    @staticmethod
    def f(x):
        return x + 1

    def test_seq_map_in_parallel(self):
        s = Seq.of(1, 2, 3)
        s = s.map(self.f, parallel_type=ParallelType.PROCESS)
        self.assertEqual(s.to_list(), [2, 3, 4])
        self.assertRaises(NotImplementedError, lambda: s.map(lambda x: x + 1, parallel_type=ParallelType.PROCESS))

    def test_seq_for_each(self):
        s = Seq.of(1, 2, 3)
        out = []
        s.for_each(lambda x: out.append(x + 1))
        self.assertEqual(out, [2, 3, 4])

    def test_seq_filter(self):
        s = Seq.of(1, 2, 3, 4, 5)
        s = s.filter(lambda x: x % 2 == 0)
        self.assertEqual(s.to_list(), [2, 4])

    def test_seq_reduce(self):
        s = Seq.of(1, 2, 3, 4, 5)
        self.assertEqual(s.reduce(lambda x, y: x + y), 15)

    def test_seq_sort(self):
        s = Seq.of(5, 3, 1, 4, 2)
        self.assertEqual(s.sort().to_list(), [1, 2, 3, 4, 5])
        self.assertEqual(s.sort(reverse=True).to_list(), [5, 4, 3, 2, 1])

        s = Seq.of([1, 9], [0, 4], [7, 6])
        self.assertEqual(s.sort(key=lambda x: x[0]).to_list(), [[0, 4], [1, 9], [7, 6]])
        self.assertEqual(s.sort(key=lambda x: x[1]).to_list(), [[0, 4], [7, 6], [1, 9]])

    def test_stream_flatten(self):
        s = Seq.of(Seq.of(1, 2, 3), Seq.of(4, 5, 6))
        self.assertEqual(s.flatten().to_list(), [1, 2, 3, 4, 5, 6])

        s = Seq.of(Seq.of(1, 2), Seq.of(3, 4), Seq.of(5, 6))
        self.assertEqual(s.flatten().to_list(), [1, 2, 3, 4, 5, 6])

        s = Seq.of(1, 2, 3).map(lambda x: Seq.of(x, x * 2)).flatten()
        self.assertEqual(s[2], 2)
        self.assertEqual(s.to_list(), [1, 2, 2, 4, 3, 6])

    def test_stream_flat_map(self):
        s = Seq.of(1, 2, 3)
        s = s.flat_map(lambda x: Seq.of(x, x * 2))
        self.assertEqual(s.to_list(), [1, 2, 2, 4, 3, 6])

    def test_stream_flatten_then_map(self):
        s = Seq.of(Seq.of(1, 2, 3), Seq.of(4, 5, 6))
        s = s.flatten().map(lambda x: x + 1)
        self.assertEqual(s.to_list(), [2, 3, 4, 5, 6, 7])

    def test_stream_flatten_then_filter(self):
        s = Seq.of(Seq.of(1, 2, 3), Seq.of(4, 5, 6))
        s = s.flatten().filter(lambda x: x % 2 == 0)
        self.assertEqual(s.to_list(), [2, 4, 6])

    def test_stream_map_then_flatten_then_map(self):
        s = Seq.of(1, 2, 3)
        s = s.map(lambda x: Seq.of(x, x * 2)).flatten().map(lambda x: x + 1)
        self.assertEqual(s.to_list(), [2, 3, 3, 5, 4, 7])

    def test_stream_skip(self):
        s = Seq.of(1, 2, 3, 4, 5)
        self.assertEqual(s.skip(2).to_list(), [3, 4, 5])