import toml


class TomlReader:

    @classmethod
    def of(cls, path: str) -> dict:
        """
        Open a toml file.

        Args:
            path (``str``): Path to the toml file.

        Returns:
            ``dict``: The opened toml file.

        """
        return toml.load(path)

    def __new__(cls, path: str):
        return cls.of(path)
