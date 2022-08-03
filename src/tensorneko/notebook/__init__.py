from tensorneko_util.notebook import ipython_available

__all__ = []

if ipython_available:
    from tensorneko_util.notebook import display
    __all__.append("display")
