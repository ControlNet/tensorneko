from __future__ import annotations
from typing import Type

from tensorneko_util.io.reader import Reader as BaseReader

try:
    from .mesh import MeshReader
    pytorch3d_available = True
except ImportError:
    MeshReader = object
    pytorch3d_available = False


class Reader(BaseReader):

    def __init__(self):
        super().__init__()
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
