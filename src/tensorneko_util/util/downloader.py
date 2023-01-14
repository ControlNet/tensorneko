from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

from tqdm.auto import tqdm


class DownloadProgressBar(tqdm):
    total: int

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dir_path: str = ".", file_path: Optional[str] = None, progress_bar: bool = True) -> str:
    """
    Download file with given URL to given directory path with progress bar.

    Args:
        url (``str``): URL of the file to download.
        dir_path (``str``, optional): Directory path to download the file to. The saved name is the same as the
            original URL name. Defaults to current directory.
        file_path (``str``, optional): File path to download the file to (override dir_path parameter).
            Default None, which uses the dir_path argument.
        progress_bar (``bool``, optional): Whether to show progress bar. Defaults to True.

    Returns:
        ``str``: File path of the downloaded file.

    """
    if file_path is not None:
        path = Path(file_path)
    else:
        path = Path(dir_path) / url.split("/")[-1]
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        if progress_bar:
            with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Downloading Marlin model") as pb:
                urlretrieve(url, filename=path, reporthook=pb.update_to)
        else:
            urlretrieve(url, filename=path)

    return str(path)

