try:
    import importlib.metadata as im

    __version__ = im.version(__package__)
except im.PackageNotFoundError:
    __version__ = "0.0.1"

from . import (
    amplitude_estimation,
    circuit_library,
    combinatorial,
    integrations,
    opt_class,
)
