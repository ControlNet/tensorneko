class TextReader:
    """TextReader for reading text file"""

    @staticmethod
    def of_plain(path: str, encoding: str = "UTF-8") -> str:
        """
        Read texts of a file.

        Args:
            path (``str``): Text file path.
            encoding (``str``, optional): File encoding. Default "UTF-8".

        Returns:
            ``str``: The texts of given file.
        """
        with open(path, "r", encoding=encoding) as file:
            text = file.read()
        return text

    of = of_plain

    def __new__(cls, path: str, encoding: str = "UTF-8") -> str:
        """Alias of :meth:`~TextReader.of"""
        return cls.of(path, encoding)
