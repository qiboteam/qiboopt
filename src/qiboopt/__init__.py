try:
    import importlib.metadata as im

    __version__ = im.version(__package__)
except Exception:
    __version__ = "0.0.1"

from . import combinatorial, integrations, opt_class
