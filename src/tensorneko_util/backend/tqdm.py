_is_tqdm_available = None
_is_mita_tqdm_available = None


def import_tqdm():
    global _is_tqdm_available
    if _is_tqdm_available is None:
        try:
            import tqdm
            _is_tqdm_available = True
            return tqdm
        except ImportError:
            _is_tqdm_available = False
            raise ImportError("tqdm is not installed. Please install it by `pip install tqdm`")
    else:
        if _is_tqdm_available:
            import tqdm
            return tqdm
        else:
            raise ImportError("tqdm is not installed. Please install it by `pip install tqdm`")


def import_tqdm_auto():
    global _is_tqdm_available
    if _is_tqdm_available is None:
        try:
            from tqdm import auto
            _is_tqdm_available = True
            return auto
        except ImportError:
            _is_tqdm_available = False
            raise ImportError("tqdm is not installed. Please install it by `pip install tqdm`")
    else:
        if _is_tqdm_available:
            from tqdm import auto
            return auto
        else:
            raise ImportError("tqdm is not installed. Please install it by `pip install tqdm`")


def import_mita_tqdm():
    global _is_mita_tqdm_available
    if _is_mita_tqdm_available is None:
        try:
            from mita_client import mita_tqdm
            _is_mita_tqdm_available = True
            return mita_tqdm
        except ImportError:
            _is_mita_tqdm_available = False
            raise ImportError("mita_client is not installed. Please install it by `pip install mita_client`")
    else:
        if _is_mita_tqdm_available:
            from mita_client import mita_tqdm
            return mita_tqdm
        else:
            raise ImportError("mita_client is not installed. Please install it by `pip install mita_client`")
