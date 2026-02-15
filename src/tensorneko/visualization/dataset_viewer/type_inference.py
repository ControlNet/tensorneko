"""
Type inference engine for the dataset viewer.

Auto-detects the type of each field in a PyTorch Dataset sample:
  "image", "text", "scalar", "tensor", "json"
"""

from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np

# Type constants
IMAGE = "image"
TEXT = "text"
SCALAR = "scalar"
TENSOR = "tensor"
JSON = "json"

# Channels that indicate image data (CHW format)
_IMAGE_CHANNELS = {1, 3, 4}
_IMAGE_MIN_SPATIAL = 8


def _is_image_shape(shape: tuple) -> bool:
    """Check if a 3D shape looks like an image in CHW format.

    Args:
        shape: A 3-element tuple (C, H, W).

    Returns:
        True if C in {1, 3, 4} and H >= 8 and W >= 8.
    """
    if len(shape) != 3:
        return False
    c, h, w = shape
    return c in _IMAGE_CHANNELS and h >= _IMAGE_MIN_SPATIAL and w >= _IMAGE_MIN_SPATIAL


def infer_field_type(value: Any) -> str:
    """Determine the type string for a single value.

    Heuristics (checked in order):
        - PIL.Image.Image → "image"
        - torch.Tensor:
            - 0-dim → "scalar"
            - 3D image-like shape → "image"
            - else → "tensor"
        - numpy.ndarray:
            - 3D image-like shape → "image"
            - else → "tensor"
        - str → "text"
        - int / float / bool → "scalar"
        - dict / list → "json"
        - fallback → "tensor"

    Args:
        value: Any Python value from a dataset sample.

    Returns:
        One of "image", "text", "scalar", "tensor", "json".
    """
    # PIL Image check (guarded import)
    try:
        from PIL import Image as PILImage

        if isinstance(value, PILImage.Image):
            return IMAGE
    except ImportError:
        pass

    # torch.Tensor
    try:
        import torch

        if isinstance(value, torch.Tensor):
            # Move to CPU for inspection if needed
            t = value.detach().cpu()
            if t.ndim == 0:
                return SCALAR
            if _is_image_shape(t.shape):
                return IMAGE
            return TENSOR
    except ImportError:
        pass

    # numpy.ndarray
    if isinstance(value, np.ndarray):
        if _is_image_shape(value.shape):
            return IMAGE
        return TENSOR

    # str must be checked before Iterable / Sized checks
    if isinstance(value, str):
        return TEXT

    # Python scalars — check bool before int (bool is subclass of int)
    if isinstance(value, (int, float, bool)):
        return SCALAR

    # Structured types
    if isinstance(value, (dict, list)):
        return JSON

    # Fallback
    return TENSOR


def normalize_sample_with_raw_keys(
    sample: Any,
) -> Tuple[Dict[str, Any], Dict[str, Hashable]]:
    if sample is None:
        return {}, {}

    if isinstance(sample, dict):
        normalized: Dict[str, Any] = {}
        raw_key_map: Dict[str, Hashable] = {}
        for key, value in sample.items():
            display_key = str(key)
            normalized[display_key] = value
            raw_key_map[display_key] = key
        return normalized, raw_key_map

    if isinstance(sample, (tuple, list)):
        normalized = {str(i): value for i, value in enumerate(sample)}
        raw_key_map = {str(i): i for i in range(len(sample))}
        return normalized, raw_key_map

    # Single value (Tensor, scalar, string, etc.)
    return {"0": sample}, {"0": 0}


def normalize_sample(sample: Any) -> Dict[str, Any]:
    """Convert any sample format to a uniform ``{str_key: value}`` dict.

    Conversion rules:
        - ``tuple`` / ``list`` → ``{"0": val0, "1": val1, ...}``
        - ``dict`` → keys converted to ``str``
        - single value (Tensor, int, str, …) → ``{"0": value}``
        - ``None`` → ``{}``

    Args:
        sample: A raw sample from a PyTorch Dataset.

    Returns:
        A dict with string keys mapping to the original field values.
    """
    normalized, _ = normalize_sample_with_raw_keys(sample)
    return normalized


def infer_schema(
    sample: Any,
    *,
    schema_override: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Inspect a dataset sample and return a field_name → type_string mapping.

    Args:
        sample: A raw sample from a PyTorch Dataset (dict, tuple, or single value).
        schema_override: Optional user-provided mapping that takes precedence over
            auto-detected types. Only the keys present in the override are replaced.

    Returns:
        A dict mapping each field name (str) to its detected type string.
        Returns an empty dict for ``None`` samples.
    """
    normalized = normalize_sample(sample)
    schema: Dict[str, str] = {}

    for key, value in normalized.items():
        schema[key] = infer_field_type(value)

    # Apply user overrides
    if schema_override:
        for key, type_str in schema_override.items():
            if key in schema:
                schema[key] = type_str

    return schema
