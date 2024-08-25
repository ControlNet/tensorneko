# clear gc for python, pytorch, etc
import gc
import torch.cuda


def run_gc():
    """
    Clear memory for Python, PyTorch, etc.
    """
    gc.collect()
    torch.cuda.empty_cache()
