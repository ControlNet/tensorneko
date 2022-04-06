import unittest

from tensorneko_util.util import Stream, Seq


class UtilStreamTest(unittest.TestCase):
    def test_stream_from_elements(self):
        s = Stream.of(1, 2, 3)
        self.assertEqual(s.to_list(), [1, 2, 3])

    def test_stream_from_iter(self):
        s = Stream([1, 2, 3])
        self.assertEqual(s.to_list(), [1, 2, 3])

    def test_stream_lshift(self):
        s = Stream.of(1, 2, 3)
        s = s << [1, 2, 3]
        self.assertEqual(s.to_list(), [1, 2, 3, 1, 2, 3])
        s = s << iter([1, 2, 3])
        self.assertEqual(s.to_list(), [1, 2, 3, 1, 2, 3, 1, 2, 3])

    def test_stream_indexing(self):
        s = Stream.of(1, 2, 3, 4, 5)
        self.assertEqual(s[1], 2)
        self.assertEqual(s[1:4].to_list(), Stream.of(2, 3, 4).to_list())

    def test_stream_print(self):
        s = Stream.of(1, 2, 3)
        self.assertEqual(str(s), "Stream(...)")
        self.assertEqual(s[1], 2)
        self.assertEqual(str(s), "Stream(1, 2, ...)")

    def test_stream_map(self):
        s = Stream.of(1, 2, 3)
        s = s.map(lambda x: x + 1)
        self.assertEqual(s.to_list(), [2, 3, 4])

    def test_stream_filter(self):
        s = Stream.of(1, 2, 3, 4, 5)
        s = s.filter(lambda x: x % 2 == 0)
        self.assertEqual(s.to_list(), [2, 4])

    def test_stream_reduce(self):
        s = Stream.of(1, 2, 3, 4, 5)
        self.assertEqual(s.reduce(lambda x, y: x + y), 15)

    def test_stream_flatten(self):
        s = Stream.of(Stream.of(1, 2, 3), Stream.of(4, 5, 6))
        self.assertEqual(s.flatten().to_list(), [1, 2, 3, 4, 5, 6])

        s = Stream.of(Stream.of(1, 2), Stream.of(3, 4), Stream.of(5, 6))
        self.assertEqual(s.flatten().to_list(), [1, 2, 3, 4, 5, 6])

        s = Stream.of(1, 2, 3).map(lambda x: Stream.of(x, x * 2)).flatten()
        self.assertEqual(s[2], 2)
        self.assertEqual(s.to_list(), [1, 2, 2, 4, 3, 6])

    def test_stream_flat_map(self):
        s = Stream.of(1, 2, 3)
        s = s.flat_map(lambda x: Stream.of(x, x * 2))
        self.assertEqual(s.to_list(), [1, 2, 2, 4, 3, 6])

    def test_stream_flatten_then_map(self):
        s = Stream.of(Stream.of(1, 2, 3), Stream.of(4, 5, 6))
        s = s.flatten().map(lambda x: x + 1)
        self.assertEqual(s.to_list(), [2, 3, 4, 5, 6, 7])

    def test_stream_flatten_then_filter(self):
        s = Stream.of(Stream.of(1, 2, 3), Stream.of(4, 5, 6))
        s = s.flatten().filter(lambda x: x % 2 == 0)
        self.assertEqual(s.to_list(), [2, 4, 6])

    def test_stream_map_then_flatten_then_map(self):
        s = Stream.of(1, 2, 3)
        s = s.map(lambda x: Stream.of(x, x * 2)).flatten().map(lambda x: x + 1)
        self.assertEqual(s.to_list(), [2, 3, 3, 5, 4, 7])

    def test_stream_skip(self):
        s = Stream.of(1, 2, 3, 4, 5)
        self.assertEqual(s.skip(2).to_list(), [3, 4, 5])

    def test_stream_take(self):
        s = Stream.of(1, 2, 3, 4, 5)
        self.assertEqual(s.take(2).to_list(), [1, 2])

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

    def test_seq_filter(self):
        s = Seq.of(1, 2, 3, 4, 5)
        s = s.filter(lambda x: x % 2 == 0)
        self.assertEqual(s.to_list(), [2, 4])

    def test_seq_reduce(self):
        s = Seq.of(1, 2, 3, 4, 5)
        self.assertEqual(s.reduce(lambda x, y: x + y), 15)

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


