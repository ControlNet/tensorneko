__all__ = []

try:
    from . import display
    __all__.append("display")
    ipython_available = True
except ImportError:
    ipython_available = False
