from __future__ import annotations

from pathlib import Path
from typing import Type, Union

from tensorneko_util.io._path_conversion import _path2str
from tensorneko_util.io.writer import Writer as BaseWriter
from .weight import WeightWriter

try:
    from .mesh import MeshWriter
    pytorch3d_available = True
except ImportError:
    MeshWriter = object
    pytorch3d_available = False


class Writer(BaseWriter):

    def __init__(self):
        super().__init__()
        self.weight = WeightWriter
        self._mesh = None

    @property
    def mesh(self) -> Type[MeshWriter]:
        if pytorch3d_available:
            if self._mesh is None:
                from .mesh import MeshWriter
                self._mesh = MeshWriter()
            return self._mesh
        else:
            raise ImportError("To use the mesh writer, please install pytorch3d.")

    def __call__(self, path: Union[str, Path], obj, *args, **kwargs):
        """Automatically infer the file type and return the corresponding result. """
        path = _path2str(path)

        if path.endswith(".pt") or path.endswith(".pth") or path.endswith(".ckpt") or path.endswith(".safetensors"):
            return self.weight(path, obj)
        else:
            return super().__call__(path, obj, *args, **kwargs)
