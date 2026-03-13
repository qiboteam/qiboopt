try:
    import importlib.metadata as im
    __version__ = im.version(__package__)
except Exception:
    __version__ = "0.0.1"

from qiboopt import combinatorial, opt_class, continuous_bandits, integrations
