import pickle


class PickleReader:

    @classmethod
    def of(cls, path: str) -> None:
        """
        Save the object to a file.

        Args:
            path (``str``): The path to the file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __new__(cls, path: str) -> None:
        """Alias to :meth:`~tensorneko_util.io.pickle.PickleReader.of`."""
        return cls.of(path)
