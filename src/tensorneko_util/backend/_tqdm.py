_is_tqdm_available = None


def import_tqdm():
    if _is_tqdm_available is None:
        global _is_tqdm_available
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
    if _is_tqdm_available is None:
        global _is_tqdm_available
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
