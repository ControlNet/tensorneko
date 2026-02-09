import unittest

import numpy as np
import torch

from tensorneko.visualization.dataset_viewer.type_inference import (
    infer_field_type,
    infer_schema,
    normalize_sample,
)


class TestNormalizeSample(unittest.TestCase):
    """Tests for normalize_sample: converts various sample formats to uniform dict."""

    def test_normalize_tuple(self):
        sample = (torch.rand(3, 32, 32), 5)
        result = normalize_sample(sample)
        self.assertIsInstance(result, dict)
        self.assertIn("0", result)
        self.assertIn("1", result)
        self.assertEqual(result["1"], 5)

    def test_normalize_dict(self):
        sample = {"image": torch.rand(3, 32, 32), "label": "cat"}
        result = normalize_sample(sample)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["label"], "cat")
        self.assertIn("image", result)

    def test_normalize_dict_non_str_keys(self):
        sample = {0: "hello", 1: 42}
        result = normalize_sample(sample)
        self.assertIn("0", result)
        self.assertIn("1", result)
        self.assertEqual(result["0"], "hello")

    def test_normalize_single_value(self):
        result = normalize_sample(42)
        self.assertEqual(result, {"0": 42})

    def test_normalize_single_tensor(self):
        t = torch.rand(3, 32, 32)
        result = normalize_sample(t)
        self.assertIn("0", result)
        self.assertTrue(torch.equal(result["0"], t))


class TestInferFieldType(unittest.TestCase):
    """Tests for infer_field_type: determines the type string for a single value."""

    def test_image_tensor_3ch(self):
        self.assertEqual(infer_field_type(torch.rand(3, 32, 32)), "image")

    def test_image_tensor_1ch(self):
        self.assertEqual(infer_field_type(torch.rand(1, 64, 64)), "image")

    def test_image_tensor_4ch(self):
        self.assertEqual(infer_field_type(torch.rand(4, 16, 16)), "image")

    def test_string_is_text(self):
        self.assertEqual(infer_field_type("hello world"), "text")

    def test_empty_string_is_text(self):
        self.assertEqual(infer_field_type(""), "text")

    def test_int_is_scalar(self):
        self.assertEqual(infer_field_type(42), "scalar")

    def test_float_is_scalar(self):
        self.assertEqual(infer_field_type(3.14), "scalar")

    def test_bool_is_scalar(self):
        self.assertEqual(infer_field_type(True), "scalar")

    def test_0d_tensor_is_scalar(self):
        self.assertEqual(infer_field_type(torch.tensor(5)), "scalar")

    def test_bool_tensor_0d_is_scalar(self):
        self.assertEqual(infer_field_type(torch.tensor(True)), "scalar")

    def test_1d_tensor_is_tensor(self):
        self.assertEqual(infer_field_type(torch.rand(16000)), "tensor")

    def test_large_3d_non_image_tensor(self):
        # channels=512 not in {1,3,4} → not image
        self.assertEqual(infer_field_type(torch.rand(512, 512, 512)), "tensor")

    def test_3d_small_spatial_not_image(self):
        # spatial dims < 8 → not image
        self.assertEqual(infer_field_type(torch.rand(3, 4, 4)), "tensor")

    def test_4d_tensor_is_tensor(self):
        self.assertEqual(infer_field_type(torch.rand(2, 3, 32, 32)), "tensor")

    def test_2d_tensor_is_tensor(self):
        self.assertEqual(infer_field_type(torch.rand(10, 10)), "tensor")

    def test_pil_image(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")
        img = Image.new("RGB", (32, 32))
        self.assertEqual(infer_field_type(img), "image")

    def test_numpy_image(self):
        arr = np.random.rand(3, 32, 32).astype(np.float32)
        self.assertEqual(infer_field_type(arr), "image")

    def test_numpy_non_image(self):
        arr = np.random.rand(100).astype(np.float32)
        self.assertEqual(infer_field_type(arr), "tensor")

    def test_nested_dict_is_json(self):
        self.assertEqual(infer_field_type({"a": {"b": 1}}), "json")

    def test_list_is_json(self):
        self.assertEqual(infer_field_type([1, 2, 3]), "json")

    def test_empty_list_is_json(self):
        self.assertEqual(infer_field_type([]), "json")

    def test_unknown_object_fallback_tensor(self):
        """Unrecognized object type → fallback to 'tensor'."""
        self.assertEqual(infer_field_type(object()), "tensor")
        self.assertEqual(infer_field_type(set()), "tensor")
        self.assertEqual(infer_field_type(b"bytes"), "tensor")


class TestInferSchema(unittest.TestCase):
    """Tests for infer_schema: returns field_name → type_string mapping for a sample."""

    def test_tuple_sample(self):
        sample = (torch.rand(3, 32, 32), 5)
        schema = infer_schema(sample)
        self.assertEqual(schema, {"0": "image", "1": "scalar"})

    def test_dict_sample(self):
        sample = {"image": torch.rand(3, 32, 32), "label": "cat"}
        schema = infer_schema(sample)
        self.assertEqual(schema, {"image": "image", "label": "text"})

    def test_single_tensor(self):
        schema = infer_schema(torch.rand(3, 32, 32))
        self.assertEqual(schema, {"0": "image"})

    def test_single_scalar(self):
        schema = infer_schema(42)
        self.assertEqual(schema, {"0": "scalar"})

    def test_mixed_types(self):
        sample = {
            "img": torch.rand(3, 64, 64),
            "caption": "a cat",
            "label": 7,
            "embedding": torch.rand(512),
            "metadata": {"source": "web"},
        }
        schema = infer_schema(sample)
        self.assertEqual(schema["img"], "image")
        self.assertEqual(schema["caption"], "text")
        self.assertEqual(schema["label"], "scalar")
        self.assertEqual(schema["embedding"], "tensor")
        self.assertEqual(schema["metadata"], "json")

    def test_user_schema_override(self):
        sample = {"image": torch.rand(3, 32, 32), "label": 5}
        user_schema = {"image": "tensor", "label": "text"}
        schema = infer_schema(sample, schema_override=user_schema)
        self.assertEqual(schema["image"], "tensor")
        self.assertEqual(schema["label"], "text")

    def test_user_schema_partial_override(self):
        sample = {"image": torch.rand(3, 32, 32), "label": 5}
        user_schema = {"label": "text"}
        schema = infer_schema(sample, schema_override=user_schema)
        self.assertEqual(schema["image"], "image")  # auto-inferred
        self.assertEqual(schema["label"], "text")  # overridden

    def test_none_sample(self):
        schema = infer_schema(None)
        self.assertEqual(schema, {})


if __name__ == "__main__":
    unittest.main()
