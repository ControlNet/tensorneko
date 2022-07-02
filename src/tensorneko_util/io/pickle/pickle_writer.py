import pickle


class PickleWriter:

    @classmethod
    def to(cls, path: str, obj: object) -> None:
        """
        Save the object to a file.

        Args:
            path (``str``): The path to the file.
            obj (``object``): The object to save.
        """
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def __new__(cls, path: str, obj: object) -> None:
        """Alias to :meth:`~tensorneko_util.io.pickle.PickleWriter.to`."""
        return cls.to(path, obj)
