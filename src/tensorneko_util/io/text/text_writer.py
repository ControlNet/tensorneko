class TextWriter:
    """TextWriter for writing files for text"""

    @staticmethod
    def to_plain(path: str, text: str, encoding: str = "UTF-8") -> None:
        """
        Save as a plain text file.

        Args:
            path (``str``): The path of output file.
            text (``str``): The content for output.
            encoding (``str``, optional): Python file IO encoding parameter. Default: "UTF-8".
        """
        with open(path, "w", encoding=encoding) as file:
            file.write(text)

    to = to_plain

    def __new__(cls, path: str, text: str, encoding: str = "UTF-8") -> None:
        """Alias of :meth:`~TextWriter.to`"""
        cls.to(path, text, encoding)
