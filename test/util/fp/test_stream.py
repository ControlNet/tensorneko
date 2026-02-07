import unittest

from tensorneko_util.backend.parallel import ParallelType
from tensorneko_util.util import Stream


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

    def test_seq_for_each(self):
        s = Stream.of(1, 2, 3)
        out = []
        s.for_each(lambda x: out.append(x + 1))
        self.assertEqual(out, [2, 3, 4])

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

    def test_from_list(self):
        s = Stream() << [1, 2, 3, 4, 5]
        self.assertEqual([1, 2, 3, 4, 5], list(s))
        self.assertEqual(2, s[1])
        self.assertEqual([1, 2], list(s[0:2]))

    def test_from_iterator(self):
        s = Stream() << range(6) << [6, 7]
        self.assertEqual([0, 1, 2, 3, 4, 5, 6, 7], list(s))

    def test_from_generator(self):
        def gen():
            yield 1
            yield 2
            yield 3

        s = Stream() << gen() << (4, 5)
        self.assertEqual(list(s), [1, 2, 3, 4, 5])

    def test_lazy_slicing(self):
        s = Stream() << range(10)
        self.assertEqual(s._cache_size, 0)

        s_slice = s[:5]
        self.assertEqual(s._cache_size, 0)
        self.assertEqual(len(list(s_slice)), 5)

    def test_lazy_slicing_recursive(self):
        s = Stream() << range(10)
        sf = s[1:3][0:2]

        self.assertEqual(s._cache_size, 0)
        self.assertEqual(len(list(sf)), 2)

    def test_fib_infinite_stream(self):
        from operator import add
        from itertools import islice

        f = Stream()
        fib = f << [0, 1] << map(add, f, islice(f, 1, None))

        self.assertEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34], list(islice(fib, 10)))
        self.assertEqual(6765, fib[20])
        self.assertEqual([832040, 1346269, 2178309, 3524578, 5702887], list(fib[30:35]))
        # 35 elements should be already evaluated
        self.assertEqual(len(fib._cache), 35)

    def test_origin_param(self):
        self.assertEqual([100], list(Stream.of(100)))
        self.assertEqual([1, 2, 3], list(Stream.of(1, 2, 3)))
        self.assertEqual(
            [1, 2, 3, 10, 20, 30], list(Stream.of(1, 2, 3) << [10, 20, 30])
        )

    def test_origin_param_string(self):
        self.assertEqual(["stream"], list(Stream.of("stream")))

    def test_stream_from_stream_and_basic_properties(self):
        source = Stream.of(1, 2, 3)
        self.assertEqual(source[0], 1)

        cloned = Stream.from_stream(source)
        self.assertEqual(cloned.to_list(), [2, 3])
        self.assertEqual(Stream.of(7, 8, 9).head, 7)
        self.assertEqual(Stream.of(7, 8, 9).tail.to_list(), [8, 9])

    def test_stream_index_errors(self):
        s = Stream.of(1, 2, 3)
        self.assertRaises(ValueError, lambda: s[-1])
        self.assertRaises(ValueError, lambda: s[0:3:0])
        self.assertRaises(TypeError, lambda: s["bad"])

    def test_stream_for_each_thread_with_progress(self):
        s = Stream.of(1, 2, 3, 4)
        out = []

        def collect(x):
            out.append(x * 2)

        s.for_each(
            collect, progress_bar=True, disable=True, parallel_type=ParallelType.THREAD
        )
        self.assertCountEqual(out, [2, 4, 6, 8])
        self.assertRaises(
            NotImplementedError,
            lambda: Stream.of(1, 2).for_each(
                lambda x: x, parallel_type=ParallelType.THREAD
            ),
        )

    def test_stream_with_for_each_returns_self(self):
        s = Stream.of(1, 2, 3)
        out = []
        result = s.with_for_each(out.append)
        self.assertIs(result, s)
        self.assertEqual(out, [1, 2, 3])

    def test_stream_repeat_and_empty_repeat(self):
        self.assertEqual(Stream.of(1, 2).repeat(3).to_list(), [1, 2, 1, 2, 1, 2])
        self.assertEqual(Stream().repeat(5).to_list(), [])

    def test_stream_map_cache_and_flatten_non_stream(self):
        mapped = Stream.of(1, 2, 3).map(lambda x: x + 10)
        self.assertEqual(mapped[2], 13)
        self.assertEqual(mapped[0], 11)

        flattened = Stream.of(Stream.of(1, 2), 3, Stream.of(4)).flatten()
        self.assertEqual(flattened.to_list(), [1, 2, 3, 4])

    def test_stream_map_random_access(self):
        """Access MapStream out of order to exercise cache hit path."""
        s = Stream.of(10, 20, 30, 40, 50).map(lambda x: x + 1)
        # First access element 3 (caches 0,1,2,3)
        self.assertEqual(s[3], 41)
        # Then access element 1 (cache hit)
        self.assertEqual(s[1], 21)

    def test_stream_filter_random_access(self):
        """Access FilterStream via indexing."""
        s = Stream.of(1, 2, 3, 4, 5, 6, 7, 8).filter(lambda x: x % 2 == 0)
        self.assertEqual(s[2], 6)
        # Cache hit
        self.assertEqual(s[0], 2)

    def test_stream_flatten_random_access(self):
        """Access FlattenStream via indexing."""
        s = Stream.of(Stream.of(1, 2), Stream.of(3, 4), Stream.of(5, 6)).flatten()
        self.assertEqual(s[4], 5)
        # Cache hit
        self.assertEqual(s[1], 2)

    def test_stream_take_random_access(self):
        """Access TakeStream out of order."""
        s = Stream.of(10, 20, 30, 40, 50).take(3)
        self.assertEqual(s[2], 30)
        self.assertEqual(s[0], 10)

    def test_stream_empty_operations(self):
        """Test operations on empty stream."""
        s = Stream()
        self.assertEqual(s.to_list(), [])
        self.assertEqual(s.map(lambda x: x + 1).to_list(), [])
        self.assertEqual(s.filter(lambda x: True).to_list(), [])
