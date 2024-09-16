__all__ = []

try:
    from . import display
    from . import animation
    __all__.extend(["display", "animation"])
    ipython_available = True
except ImportError:
    ipython_available = False
