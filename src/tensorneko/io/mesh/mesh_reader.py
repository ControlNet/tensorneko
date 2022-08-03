import pytorch3d


class MeshReader:

    @classmethod
    def of(cls):
        raise NotImplementedError()

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError()
