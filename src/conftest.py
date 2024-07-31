"""Configure doctesting by pytest."""

import numpy as np
import pytest

import curvecluster

np.random.seed(0)


@pytest.fixture(autouse=True)
def doctest_pre_code(doctest_namespace):
    """Import modules for doctesting.

    This fixture is equivalent to::

        import numpy as np
        from curvecluster import *
    """
    doctest_namespace["np"] = np
    for var in curvecluster.__all__:
        doctest_namespace[var] = getattr(curvecluster, var)
