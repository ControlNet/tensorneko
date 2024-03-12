from pathlib import Path
from typing import Optional, Union, List
from urllib.request import urlretrieve
from threading import Thread

from ..backend.tqdm import import_tqdm_auto

try:
    auto = import_tqdm_auto()
    tqdm = auto.tqdm
    _is_progress_bar_available = True
except ImportError:
    _is_progress_bar_available = False
    DownloadProgressBar = object
else:
    class DownloadProgressBar(tqdm):
        total: int

        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)


def download_file(url: str, dir_path: str = ".", file_path: Optional[str] = None,
    progress_bar: Union[bool, int] = True
) -> str:
    """
    Download file with given URL to given directory path with progress bar.

    Args:
        url (``str``): URL of the file to download.
        dir_path (``str``, optional): Directory path to download the file to. The saved name is the same as the
            original URL name. Defaults to current directory.
        file_path (``str``, optional): File path to download the file to (override dir_path parameter).
            Default None, which uses the dir_path argument.
        progress_bar (``bool`` | ``int``, optional): Whether to show progress bar. Defaults to True.
            True means one progress bar for the whole download process.
            False means no progress bar.
            Any int means enabling the progress bar and it is the position of the progress bar.

    Returns:
        ``str``: File path of the downloaded file.

    """
    if file_path is not None:
        path = Path(file_path)
    else:
        path = Path(dir_path) / url.split("/")[-1]
    path.parent.mkdir(exist_ok=True, parents=True)

    if type(progress_bar) is not False:
        if not _is_progress_bar_available:
            raise ImportError("Please install tqdm to use progress bar.")

        position = None if type(progress_bar) is bool and progress_bar is True else progress_bar
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1,
            desc=f"Downloading {path.name}", position=position
        ) as pb:
            urlretrieve(url, filename=path, reporthook=pb.update_to)
    else:
        urlretrieve(url, filename=path)

    return str(path)


def download_file_thread(url: str, dir_path: str = ".", file_path: Optional[str] = None,
    progress_bar: Union[bool, int] = True
) -> Thread:
    """
    Download file with given URL to given directory path with progress bar.

    Args:
        url (``str``): URL of the file to download.
        dir_path (``str``, optional): Directory path to download the file to. The saved name is the same as the
            original URL name. Defaults to current directory.
        file_path (``str``, optional): File path to download the file to (override dir_path parameter).
            Default None, which uses the dir_path argument.
        progress_bar (``bool`` | ``int``, optional): Whether to show progress bar. Defaults to True.
            True means one progress bar for the whole download process.
            False means no progress bar.
            Any int means enabling the progress bar and it is the position of the progress bar.

    Returns:
        ``Thread``: The thread of downloading.

    """
    return Thread(target=download_file, args=(url, dir_path, file_path, progress_bar))


def download_files_thread(urls: List[str], dir_path: str = ".", file_paths: Optional[List[str]] = None,
    progress_bar: Union[bool, int] = True
) -> List[Thread]:
    """
    Download files with given URLs to given directory path with progress bar.

    Args:
        urls (``List[str]``): URLs of the files to download.
        dir_path (``str``, optional): Directory path to download the file to. The saved name is the same as the
            original URL name. Defaults to current directory.
        file_paths (``List[str]``, optional): File paths to download the files to (override dir_path parameter).
            Default None, which uses the dir_path argument.
        progress_bar (``bool`` | ``int``, optional): Whether to show progress bar. Defaults to True.

    Returns:
        ``List[Thread]``: The threads of downloading.

    """
    if progress_bar:
        progress_bars = list(range(len(urls)))
    else:
        progress_bars = [False] * len(urls)

    if file_paths is None:
        file_paths = [None] * len(urls)

    return [
        download_file_thread(url, dir_path, file_path, pb)
        for url, file_path, pb in zip(urls, file_paths, progress_bars)
    ]
