from __future__ import annotations

from typing import Type

from tensorneko_util.io.writer import Writer as BaseWriter

try:
    from .mesh import MeshWriter
    pytorch3d_available = True
except ImportError:
    MeshWriter = object
    pytorch3d_available = False


class Writer(BaseWriter):

    def __init__(self):
        super().__init__()
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
