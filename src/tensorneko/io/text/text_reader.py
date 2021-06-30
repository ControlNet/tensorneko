class TextReader:
    @staticmethod
    def of(path: str, encoding: str = "UTF-8") -> str:
        with open(path, "r", encoding=encoding) as file:
            text = file.read()
        return text
