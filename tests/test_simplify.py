import numpy as np
import pytest

from curvecluster import simplify_polyline


def test_simplify():
    P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    P_l2, thres_l2 = simplify_polyline(P, 2)
    _, thres_l4 = simplify_polyline(P, 4)
    assert np.all(P_l2 == P[[0, -1]])
    assert thres_l2 >= thres_l4


def test_simplify_failure():
    P = np.array([[0, 0], [2, 2], [4, 2], [4, 4], [2, 1], [5, 1], [7, 2]])
    with pytest.raises(RuntimeError):
        simplify_polyline(P, 3)
