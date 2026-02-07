import unittest

from tensorneko_util.util.bimap import BiMap


class BiMapTest(unittest.TestCase):
    def test_bimap_creation(self):
        bimap = BiMap()
        self.assertEqual(len(bimap), 0)

        bimap = BiMap({1: "a", 2: "b"})
        self.assertEqual(len(bimap), 2)
        self.assertEqual(bimap.forward, {1: "a", 2: "b"})
        self.assertEqual(bimap.backward, {"a": 1, "b": 2})

    def test_bimap_getitem(self):
        bimap = BiMap({1: "a", 2: "b"})
        self.assertEqual(bimap[1], "a")
        self.assertEqual(bimap.get(1), "a")
        self.assertEqual(bimap.get_key("a"), 1)
        self.assertEqual(bimap.get_key("b"), 2)
        self.assertEqual(bimap.get(3, default="c"), "c")
        self.assertEqual(bimap.get_key("c", default=3), 3)

        with self.assertRaises(KeyError):
            bimap[3]
        with self.assertRaises(KeyError):
            bimap.get(3)
        with self.assertRaises(KeyError):
            bimap.get_key("c")

    def test_bimap_setitem(self):
        bimap = BiMap()
        bimap[1] = "a"
        self.assertEqual(len(bimap), 1)
        self.assertEqual(bimap.forward, {1: "a"})
        self.assertEqual(bimap.backward, {"a": 1})

        bimap[2] = "b"
        self.assertEqual(len(bimap), 2)
        self.assertEqual(bimap.forward, {1: "a", 2: "b"})
        self.assertEqual(bimap.backward, {"a": 1, "b": 2})

        bimap[3] = "c"
        self.assertEqual(len(bimap), 3)
        self.assertEqual(bimap.forward, {1: "a", 2: "b", 3: "c"})
        self.assertEqual(bimap.backward, {"a": 1, "b": 2, "c": 3})

        bimap[4] = "d"
        self.assertEqual(len(bimap), 4)
        self.assertEqual(bimap.forward, {1: "a", 2: "b", 3: "c", 4: "d"})
        self.assertEqual(bimap.backward, {"a": 1, "b": 2, "c": 3, "d": 4})

    def test_bimap_delitem(self):
        bimap = BiMap({1: "a", 2: "b"})
        del bimap[1]
        self.assertEqual(len(bimap), 1)
        self.assertEqual(bimap.forward, {2: "b"})
        self.assertEqual(bimap.backward, {"b": 2})

        del bimap[2]
        self.assertEqual(len(bimap), 0)
        self.assertEqual(bimap.forward, {})
        self.assertEqual(bimap.backward, {})

        with self.assertRaises(KeyError):
            del bimap[3]

    def test_bimap_clear(self):
        bimap = BiMap({1: "a", 2: "b"})
        bimap.clear()
        self.assertEqual(len(bimap), 0)
        self.assertEqual(bimap.forward, {})
        self.assertEqual(bimap.backward, {})

    def test_bimap_contains(self):
        bimap = BiMap({1: "a", 2: "b"})
        self.assertTrue(1 in bimap)
        self.assertTrue("a" in bimap.backward)
        self.assertTrue(2 in bimap)
        self.assertTrue("b" in bimap.backward)
        self.assertFalse(3 in bimap)
        self.assertFalse("c" in bimap.backward)

    def test_bimap_len(self):
        bimap = BiMap()
        self.assertEqual(len(bimap), 0)
        bimap[1] = "a"
        self.assertEqual(len(bimap), 1)
        bimap[2] = "b"
        self.assertEqual(len(bimap), 2)
        del bimap[1]
        self.assertEqual(len(bimap), 1)
        del bimap[2]
        self.assertEqual(len(bimap), 0)

    def test_bimap_iter(self):
        bimap = BiMap({1: "a", 2: "b"})
        self.assertEqual(list(bimap), [1, 2])
        self.assertEqual(list(bimap.backward), ["a", "b"])

    def test_bimap_copy_items_keys_values(self):
        bimap = BiMap({1: "a", 2: "b"})
        copied = bimap.__copy__()
        self.assertIsNone(copied)
        self.assertEqual(list(bimap.items()), [(1, "a"), (2, "b")])
        self.assertEqual(list(bimap.keys()), [1, 2])
        self.assertEqual(list(bimap.values()), ["a", "b"])

    def test_bimap_update_with_dict_and_bimap(self):
        bimap = BiMap({1: "a"})
        bimap.update({2: "b"})
        self.assertEqual(bimap.forward, {1: "a", 2: "b"})
        self.assertEqual(bimap.backward, {"a": 1, "b": 2})

        other = BiMap({3: "c"})
        bimap.update(other)
        self.assertEqual(bimap.forward, {1: "a", 2: "b", 3: "c"})
        self.assertEqual(bimap.backward, {"a": 1, "b": 2, "c": 3})

    def test_bimap_duplicate_value_error_case(self):
        bimap = BiMap({1: "x"})
        bimap[2] = "x"
        self.assertEqual(bimap.get_key("x"), 2)

        del bimap[1]
        self.assertEqual(bimap.forward, {2: "x"})
        with self.assertRaises(KeyError):
            bimap.get_key("x")

    def test_bimap_pop_and_popitem(self):
        bimap = BiMap({1: "a", 2: "b", 3: "c"})
        popped = bimap.pop(2)
        self.assertEqual(popped, "b")
        self.assertEqual(bimap.forward, {1: "a", 3: "c"})
        self.assertEqual(bimap.backward, {"a": 1, "c": 3})

        key, value = bimap.popitem()
        self.assertFalse((key, value) == (2, "b"))
        self.assertTrue(key in [1, 3])
        self.assertTrue(value in ["a", "c"])
        self.assertTrue(value not in bimap.backward)

    def test_bimap_repr_str_eq_hash_and_bool(self):
        bimap = BiMap({1: "a"})
        self.assertEqual(repr(bimap), "BiMap({1: 'a'})")
        self.assertEqual(str(bimap), "BiMap({1: 'a'})")
        self.assertTrue(bimap == BiMap({1: "a"}))
        self.assertFalse(bimap == {1: "a"})
        self.assertTrue(bool(bimap))
        self.assertFalse(bool(BiMap()))

        with self.assertRaises(TypeError):
            hash(bimap)
