from __future__ import annotations

from pathlib import Path
from typing import Type, Union, Any

from tensorneko_util.io._path_conversion import _path2str
from tensorneko_util.io.reader import Reader as BaseReader
from .weight import WeightReader

try:
    from .mesh import MeshReader
    pytorch3d_available = True
except ImportError:
    MeshReader = object
    pytorch3d_available = False


class Reader(BaseReader):

    def __init__(self):
        super().__init__()
        self.weight = WeightReader
        self._mesh = None

    @property
    def mesh(self) -> Type[MeshReader]:
        if pytorch3d_available:
            if self._mesh is None:
                from .mesh import MeshReader
                self._mesh = MeshReader
            return self._mesh
        else:
            raise ImportError("To use the mesh reader, please install pytorch3d.")

    def __call__(self, path: Union[str, Path], *args, **kwargs) -> Any:
        """Automatically infer the file type and return the corresponding result. """
        path = _path2str(path)
        if path.endswith(".pt") or path.endswith(".pth") or path.endswith(".ckpt") or path.endswith(".safetensors"):
            return self.weight(path, *args, **kwargs)
        else:
            return super().__call__(path, *args, **kwargs)
