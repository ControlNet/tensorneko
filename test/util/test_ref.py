import unittest

from tensorneko_util.util.ref import BoolRef, FloatRef, IntRef, Ref, StringRef, ref


class _DummyComponent:
    def __init__(self):
        self.update_count = 0

    def update_view(self):
        self.update_count += 1


class UtilRefTest(unittest.TestCase):
    def test_ref_stores_and_retrieves_value(self):
        value_ref = Ref(10)
        self.assertEqual(value_ref.value, 10)

    def test_ref_value_setter_updates_value(self):
        value_ref = Ref("before")
        value_ref.value = "after"
        self.assertEqual(value_ref.value, "after")

    def test_ref_value_setter_updates_bound_component(self):
        value_ref = Ref(1)
        comp = _DummyComponent()
        value_ref.bound_comp = comp

        value_ref.value = 2
        value_ref.value = 3

        self.assertEqual(comp.update_count, 2)

    def test_ref_apply_and_rshift(self):
        value_ref = Ref(3)
        self.assertIs(value_ref.apply(lambda x: x + 2), value_ref)
        self.assertEqual(value_ref.value, 5)

        value_ref >> (lambda x: x * 4)
        self.assertEqual(value_ref.value, 20)

    def test_ref_apply_raises_on_type_mismatch(self):
        value_ref = Ref(1)
        with self.assertRaises(AssertionError):
            value_ref.apply(lambda _: "bad")

    def test_ref_string_repr_and_none(self):
        string_ref = StringRef("hello")
        none_ref = Ref(None)

        self.assertEqual(str(string_ref), "hello")
        self.assertIn("StringRef", repr(string_ref))
        self.assertIsNone(none_ref.value)

    def test_ref_with_different_types(self):
        int_ref = ref(7)
        float_ref = ref(1.5)
        bool_ref = ref(True)
        string_ref = ref("abc")
        list_ref = Ref([1, 2, 3])

        self.assertIsInstance(int_ref, IntRef)
        self.assertIsInstance(float_ref, FloatRef)
        self.assertIsInstance(bool_ref, BoolRef)
        self.assertIsInstance(string_ref, StringRef)
        self.assertEqual(int(int_ref), 7)
        self.assertEqual(float(float_ref), 1.5)
        self.assertTrue(bool(bool_ref))
        self.assertEqual(list_ref.value, [1, 2, 3])
