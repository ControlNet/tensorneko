import unittest

from tensorneko_util.util.bimap import BiMap


class BiMapTest(unittest.TestCase):

    def test_bimap_creation(self):
        bimap = BiMap()
        self.assertEqual(len(bimap), 0)

        bimap = BiMap({1: 'a', 2: 'b'})
        self.assertEqual(len(bimap), 2)
        self.assertEqual(bimap.forward, {1: 'a', 2: 'b'})
        self.assertEqual(bimap.backward, {'a': 1, 'b': 2})

    def test_bimap_getitem(self):
        bimap = BiMap({1: 'a', 2: 'b'})
        self.assertEqual(bimap[1], 'a')
        self.assertEqual(bimap.get(1), 'a')
        self.assertEqual(bimap.get_key('a'), 1)
        self.assertEqual(bimap.get_key('b'), 2)
        self.assertEqual(bimap.get(3, default='c'), 'c')
        self.assertEqual(bimap.get_key('c', default=3), 3)

        with self.assertRaises(KeyError):
            bimap[3]
        with self.assertRaises(KeyError):
            bimap.get(3)
        with self.assertRaises(KeyError):
            bimap.get_key('c')

    def test_bimap_setitem(self):
        bimap = BiMap()
        bimap[1] = 'a'
        self.assertEqual(len(bimap), 1)
        self.assertEqual(bimap.forward, {1: 'a'})
        self.assertEqual(bimap.backward, {'a': 1})

        bimap[2] = 'b'
        self.assertEqual(len(bimap), 2)
        self.assertEqual(bimap.forward, {1: 'a', 2: 'b'})
        self.assertEqual(bimap.backward, {'a': 1, 'b': 2})

        bimap[3] = 'c'
        self.assertEqual(len(bimap), 3)
        self.assertEqual(bimap.forward, {1: 'a', 2: 'b', 3: 'c'})
        self.assertEqual(bimap.backward, {'a': 1, 'b': 2, 'c': 3})

        bimap[4] = 'd'
        self.assertEqual(len(bimap), 4)
        self.assertEqual(bimap.forward, {1: 'a', 2: 'b', 3: 'c', 4: 'd'})
        self.assertEqual(bimap.backward, {'a': 1, 'b': 2, 'c': 3, 'd': 4})

    def test_bimap_delitem(self):
        bimap = BiMap({1: 'a', 2: 'b'})
        del bimap[1]
        self.assertEqual(len(bimap), 1)
        self.assertEqual(bimap.forward, {2: 'b'})
        self.assertEqual(bimap.backward, {'b': 2})

        del bimap[2]
        self.assertEqual(len(bimap), 0)
        self.assertEqual(bimap.forward, {})
        self.assertEqual(bimap.backward, {})

        with self.assertRaises(KeyError):
            del bimap[3]

    def test_bimap_clear(self):
        bimap = BiMap({1: 'a', 2: 'b'})
        bimap.clear()
        self.assertEqual(len(bimap), 0)
        self.assertEqual(bimap.forward, {})
        self.assertEqual(bimap.backward, {})

    def test_bimap_contains(self):
        bimap = BiMap({1: 'a', 2: 'b'})
        self.assertTrue(1 in bimap)
        self.assertTrue('a' in bimap.backward)
        self.assertTrue(2 in bimap)
        self.assertTrue('b' in bimap.backward)
        self.assertFalse(3 in bimap)
        self.assertFalse('c' in bimap.backward)

    def test_bimap_len(self):
        bimap = BiMap()
        self.assertEqual(len(bimap), 0)
        bimap[1] = 'a'
        self.assertEqual(len(bimap), 1)
        bimap[2] = 'b'
        self.assertEqual(len(bimap), 2)
        del bimap[1]
        self.assertEqual(len(bimap), 1)
        del bimap[2]
        self.assertEqual(len(bimap), 0)

    def test_bimap_iter(self):
        bimap = BiMap({1: 'a', 2: 'b'})
        self.assertEqual(list(bimap), [1, 2])
        self.assertEqual(list(bimap.backward), ['a', 'b'])
