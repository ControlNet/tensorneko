import pandas as pd


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

    of_json = pd.read_json
    of_csv = pd.read_csv
    of_xml = pd.read_xml
    of = of_plain
