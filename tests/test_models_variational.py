"""
Testing Variational Quantum Circuits.
"""

import pathlib

import numpy as np

REGRESSION_FOLDER = pathlib.Path(__file__).with_name("regressions")


def assert_regression_fixture(backend, array, filename, rtol=1e-5, atol=1e-12):
    """Check array matches data inside filename.

    Args:
        array: numpy array/
        filename: fixture filename

    If filename does not exists, this function
    creates the missing file otherwise it loads
    from file and compare.
    """

    def load(filename):
        return np.loadtxt(filename)

    filename = REGRESSION_FOLDER / filename
    try:
        array_fixture = load(filename)
    except:  # pragma: no cover
        # case not tested in GitHub workflows because files exist
        np.savetxt(filename, array)
        array_fixture = load(filename)
    backend.assert_allclose(array, array_fixture, rtol=rtol, atol=atol)
