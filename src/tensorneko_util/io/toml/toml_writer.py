import toml


class TomlWriter:

    @classmethod
    def to(cls, path: str, obj: dict):
        """
        Save as Toml file from a dictionary.

        Args:
            path (``str``): The path of output file.
            obj (``dict``): The toml data which need to be used for output.
        """
        with open(path, "w", encoding="UTF-8") as f:
            toml.dump(obj, f)

    @classmethod
    def __new__(cls, path: str, obj: dict):
        """Alias of :meth:`~tensorneko.io.toml.toml_writer.TomlWriter.to`."""
        cls.to(path, obj)
