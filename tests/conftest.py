import numpy as np
import pytest

P_VERT = [[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]]


@pytest.fixture
def P_vert():
    return np.array(P_VERT)
