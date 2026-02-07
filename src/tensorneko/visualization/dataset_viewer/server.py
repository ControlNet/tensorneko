"""
FastAPI-based dataset viewer server.

Provides a web interface and JSON API for browsing PyTorch Dataset samples,
with automatic type inference and media rendering.
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Dict, Optional

import fastapi
import uvicorn
from fastapi.responses import FileResponse, JSONResponse, Response

from .type_inference import infer_schema, normalize_sample
from .renderers import render_media, render_metadata, get_media_content_type

_WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


class DatasetVisualizer:
    """Interactive dataset viewer backed by a FastAPI server.

    Parameters
    ----------
    dataset :
        A PyTorch map-style Dataset (must have ``__len__`` and ``__getitem__``).
    schema : dict, optional
        User-provided ``{field_name: type_string}`` override.  If *None*,
        the schema is auto-inferred from the first sample.
    page_size : int
        Default page size for the ``/api/samples`` endpoint.
    """

    def __init__(
        self,
        dataset: Any,
        schema: Optional[Dict[str, str]] = None,
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

        # Infer or accept schema
        if schema is not None:
            self._schema = schema
        elif len(dataset) > 0:
            try:
                self._schema = infer_schema(dataset[0])
            except Exception:
                self._schema: Dict[str, str] = {}
        else:
            self._schema: Dict[str, str] = {}

        # Build FastAPI app
        self.app = fastapi.FastAPI()
        self._register_routes()

        # Lifecycle state
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

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
                except Exception:
                    continue  # skip errored samples
                fields_dict = normalize_sample(raw)
                field_meta = {}
                for name, value in fields_dict.items():
                    ftype = self._schema.get(name, "tensor")
                    field_meta[name] = render_metadata(value, ftype)
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
            fields_dict = normalize_sample(raw)
            field_meta = {}
            for name, value in fields_dict.items():
                ftype = self._schema.get(name, "tensor")
                field_meta[name] = render_metadata(value, ftype)
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
        assert self._server is not None
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._server.serve())

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
