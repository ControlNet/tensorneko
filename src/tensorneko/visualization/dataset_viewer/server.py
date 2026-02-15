"""
FastAPI-based dataset viewer server.

Provides a web interface and JSON API for browsing PyTorch Dataset samples,
with automatic type inference and media rendering.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
import logging
import os
import threading
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Union

import fastapi
import uvicorn
from fastapi.responses import FileResponse, JSONResponse, Response

from .type_inference import (
    infer_schema,
    normalize_sample,
    normalize_sample_with_raw_keys,
)
from .renderers import render_media, render_metadata, get_media_content_type

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
_logger = logging.getLogger(__name__)

_FieldLabelMapping = Union[Sequence[str], Mapping[Any, str]]


class DatasetVisualizer:
    """Interactive dataset viewer backed by a FastAPI server.

    Parameters
    ----------
    dataset :
        A PyTorch map-style Dataset (must have ``__len__`` and ``__getitem__``).
    schema : dict, optional
        User-provided ``{field_name: type_string}`` override.  If *None*,
        the schema is auto-inferred from the first sample.
    label_mappings : dict, optional
        Optional mapping from raw field key to label names. Keys can be tuple
        indices (e.g. ``1``) or dict field keys (e.g. ``"label"``).
        Values can be either ``list[str]`` or ``dict[index, name]``.
    page_size : int
        Default page size for the ``/api/samples`` endpoint.
    """

    def __init__(
        self,
        dataset: Any,
        schema: Optional[Dict[str, str]] = None,
        label_mappings: Optional[Mapping[Hashable, _FieldLabelMapping]] = None,
        page_size: int = 20,
    ) -> None:
        # Validate map-style dataset
        if not (hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
            raise TypeError(
                "DatasetVisualizer requires a map-style Dataset with "
                "__len__ and __getitem__. IterableDataset is not supported."
            )

        self._dataset = dataset
        self._page_size = page_size
        self._label_mappings: Dict[Hashable, _FieldLabelMapping] = (
            dict(label_mappings) if label_mappings is not None else {}
        )

        # Infer or accept schema
        if schema is not None:
            self._schema = schema
        elif len(dataset) > 0:
            try:
                self._schema = infer_schema(dataset[0])
            except Exception as e:
                _logger.warning(
                    "Failed to infer dataset schema from sample 0: %s",
                    e,
                )
                self._schema: Dict[str, str] = {}
        else:
            self._schema: Dict[str, str] = {}

        # Build FastAPI app
        self.app = fastapi.FastAPI()
        self._register_routes()

        # Lifecycle state
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def _find_field_mapping(
        self,
        field_name: str,
        raw_key: Hashable,
    ) -> Optional[_FieldLabelMapping]:
        candidates = [raw_key, field_name]

        raw_key_str = str(raw_key)
        if raw_key_str not in candidates:
            candidates.append(raw_key_str)

        for key in candidates:
            if key in self._label_mappings:
                return self._label_mappings[key]

        return None

    @staticmethod
    def _coerce_label_index(value: Any) -> Optional[int]:
        try:
            import numpy as np

            if isinstance(value, np.generic):
                value = value.item()
        except ImportError:
            pass

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        if isinstance(value, float) and value.is_integer():
            return int(value)

        return None

    @staticmethod
    def _tensor_label_index(value: Any) -> Optional[int]:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu()
                if tensor.ndim != 1 or tensor.numel() == 0:
                    return None
                return int(torch.argmax(tensor.float()).item())
        except ImportError:
            pass

        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                if value.ndim != 1 or value.size == 0:
                    return None
                return int(np.argmax(value))
        except ImportError:
            pass
        except ValueError:
            return None
        except TypeError:
            return None

        return None

    @staticmethod
    def _resolve_label_name(mapping: _FieldLabelMapping, index: int) -> Optional[str]:
        if isinstance(mapping, MappingABC):
            if index in mapping:
                return str(mapping[index])

            str_index = str(index)
            if str_index in mapping:
                return str(mapping[str_index])

            return None

        if isinstance(mapping, SequenceABC) and not isinstance(mapping, (str, bytes)):
            if 0 <= index < len(mapping):
                return str(mapping[index])

        return None

    def _append_label_metadata(
        self,
        field_name: str,
        raw_key: Hashable,
        value: Any,
        field_type: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        mapping = self._find_field_mapping(field_name, raw_key)
        if mapping is None:
            return metadata

        index: Optional[int] = None
        if field_type == "scalar":
            index = self._coerce_label_index(metadata.get("value"))
        elif field_type == "tensor":
            index = self._tensor_label_index(value)

        if index is None:
            return metadata

        label_name = self._resolve_label_name(mapping, index)
        if label_name is None:
            return metadata

        metadata = dict(metadata)
        metadata["label"] = label_name
        metadata["label_index"] = index
        return metadata

    # ------------------------------------------------------------------
    # Route registration
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        @app.get("/")
        def root():
            index_path = os.path.join(_WEB_DIR, "index.html")
            return FileResponse(index_path, media_type="text/html")

        @app.get("/api/info")
        def api_info():
            return {
                "length": len(self._dataset),
                "schema": self._schema,
                "page_size": self._page_size,
            }

        @app.get("/api/samples")
        def api_samples(offset: int = 0, limit: int = self._page_size):
            length = len(self._dataset)
            end = min(offset + limit, length)
            results = []
            for i in range(offset, end):
                try:
                    raw = self._dataset[i]
                except Exception as e:
                    _logger.warning("Failed to load dataset sample %d: %s", i, e)
                    continue
                fields_dict, raw_key_map = normalize_sample_with_raw_keys(raw)
                field_meta = {}
                for name, value in fields_dict.items():
                    ftype = self._schema.get(name, "tensor")
                    metadata = render_metadata(value, ftype)
                    raw_key = raw_key_map.get(name, name)
                    field_meta[name] = self._append_label_metadata(
                        name,
                        raw_key,
                        value,
                        ftype,
                        metadata,
                    )
                results.append({"idx": i, "fields": field_meta})
            return results

        @app.get("/api/sample/{idx}")
        def api_sample(idx: int):
            length = len(self._dataset)
            if idx < 0 or idx >= length:
                return JSONResponse(
                    {"error": f"Index {idx} out of range [0, {length})"},
                    status_code=404,
                )
            try:
                raw = self._dataset[idx]
            except Exception as e:
                return JSONResponse(
                    {"error": str(e)},
                    status_code=500,
                )
            fields_dict, raw_key_map = normalize_sample_with_raw_keys(raw)
            field_meta = {}
            for name, value in fields_dict.items():
                ftype = self._schema.get(name, "tensor")
                metadata = render_metadata(value, ftype)
                raw_key = raw_key_map.get(name, name)
                field_meta[name] = self._append_label_metadata(
                    name,
                    raw_key,
                    value,
                    ftype,
                    metadata,
                )
            return {"idx": idx, "fields": field_meta}

        @app.get("/media/{idx}/{field}")
        def media(idx: int, field: str):
            length = len(self._dataset)
            if idx < 0 or idx >= length:
                return JSONResponse(
                    {"error": f"Index {idx} out of range"},
                    status_code=404,
                )
            ftype = self._schema.get(field)
            if ftype is None or ftype != "image":
                return JSONResponse(
                    {"error": f"Field '{field}' is not a media type"},
                    status_code=404,
                )
            try:
                raw = self._dataset[idx]
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            fields_dict = normalize_sample(raw)
            if field not in fields_dict:
                return JSONResponse(
                    {"error": f"Field '{field}' not found"},
                    status_code=404,
                )
            value = fields_dict[field]
            media_bytes = render_media(value, ftype)
            content_type = get_media_content_type(ftype)
            return Response(content=media_bytes, media_type=content_type)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def serve(self, port: int = 8000, host: str = "127.0.0.1") -> None:
        """Start the server in blocking mode."""
        uvicorn.run(self.app, host=host, port=port, log_level="warning")

    def start(self, port: int = 8000, host: str = "127.0.0.1") -> None:
        """Start the server in a background daemon thread."""
        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        print(f"Dataset viewer started at http://{host}:{port}/")

    def _run_server(self) -> None:
        if self._server is None:
            raise RuntimeError("Server has not been initialized. Call start() first.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._server.serve())
        finally:
            # Let pending tasks (e.g. starlette lifespan) finish cleanly
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def stop(self) -> None:
        """Signal shutdown and wait for the server thread to exit."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    def __enter__(self) -> "DatasetVisualizer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
