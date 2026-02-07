"""
Renderers for the dataset viewer.

Converts dataset field values to display-friendly formats:
- Images → PNG bytes (in-memory)
- Text / scalars / tensors / JSON → JSON-serializable dicts
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import numpy as np


def _ensure_pil():
    """Raise a helpful ImportError when PIL is not installed."""
    try:
        from PIL import Image  # noqa: F401

        return Image
    except ImportError:
        raise ImportError(
            "PIL is required for image rendering. Install it with: pip install Pillow"
        )


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch.Tensor or numpy array to a CPU numpy array."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return value
    raise TypeError(f"Cannot convert {type(value)} to numpy array")


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    """
    Convert a CHW numpy array to PNG bytes.

    Handles:
    - Float arrays [0, 1] → uint8 [0, 255]
    - uint8 arrays → pass-through
    - 1-channel grayscale (1, H, W) → squeeze to (H, W)
    - 3-channel RGB or 4-channel RGBA → HWC transpose
    """
    Image = _ensure_pil()

    # CHW → HWC (or squeeze for grayscale)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)  # (1, H, W) → (H, W)
    elif arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) → (H, W, C)

    # Normalize to uint8
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)

    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_media(value: Any, field_type: str) -> bytes:
    """
    Render a field value as raw media bytes (currently images only).

    Parameters
    ----------
    value : Any
        A torch.Tensor, numpy.ndarray, or PIL.Image.Image.
    field_type : str
        The semantic type of the field (e.g. ``"image"``).

    Returns
    -------
    bytes
        PNG-encoded image bytes.
    """
    if field_type == "image":
        Image = _ensure_pil()

        # PIL Image → save directly
        if isinstance(value, Image.Image):
            buf = BytesIO()
            value.save(buf, format="PNG")
            return buf.getvalue()

        # Tensor or ndarray → convert
        arr = _to_numpy(value)
        return _array_to_png_bytes(arr)

    raise ValueError(f"render_media does not support field_type={field_type!r}")


def render_metadata(value: Any, field_type: str) -> Dict[str, Any]:
    """
    Render a field value as a JSON-serializable metadata dict.

    Parameters
    ----------
    value : Any
        The raw field value.
    field_type : str
        One of ``"image"``, ``"text"``, ``"scalar"``, ``"tensor"``,
        ``"json"``, or an arbitrary string (falls back to ``"unknown"``).

    Returns
    -------
    dict
        A JSON-serializable dictionary describing the value.
    """
    if field_type == "image":
        return {"type": "image"}

    if field_type == "text":
        return {"type": "text", "value": str(value)}

    if field_type == "scalar":
        # Convert tensor scalars to Python primitives
        try:
            import torch

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
        except ImportError:
            pass
        return {"type": "scalar", "value": value}

    if field_type == "tensor":
        try:
            import torch

            if isinstance(value, torch.Tensor):
                t = value.detach().cpu().float()
                data = value.detach().cpu().tolist() if value.numel() < 100 else None
                return {
                    "type": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(t.min()),
                    "max": float(t.max()),
                    "mean": float(t.mean()),
                    "data": data,
                }
        except ImportError:
            pass

        if isinstance(value, np.ndarray):
            data = value.tolist() if value.size < 100 else None
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(value.min()),
                "max": float(value.max()),
                "mean": float(value.mean()),
                "data": data,
            }

        return {"type": "unknown", "value": str(value)}

    if field_type == "json":
        return {"type": "json", "value": value}

    # Default / unknown
    return {"type": "unknown", "value": str(value)}


def get_media_content_type(field_type: str) -> str:
    """
    Return the MIME content-type for a given field type.

    Parameters
    ----------
    field_type : str
        The semantic type (e.g. ``"image"``).

    Returns
    -------
    str
        A MIME type string.
    """
    if field_type == "image":
        return "image/png"
    return "application/octet-stream"
