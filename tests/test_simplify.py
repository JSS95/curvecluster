import numpy as np

from curvecluster._frechet import fd
from curvecluster.simplify import FS, simplify_polyline


def test_simplify(P_vert):
    P_l2, thres_l2 = simplify_polyline(P_vert, 2)
    _, thres_l4 = simplify_polyline(P_vert, 4)
    assert np.all(P_l2 == P_vert[[0, -1]])
    assert thres_l2 >= thres_l4


def test_FS(P_vert):
    def check(P, epsilon):
        simp_idx = FS(P, epsilon)
        assert np.all(
            [
                fd(
                    P[[simp_idx[i], simp_idx[i + 1]]],
                    P[simp_idx[i] : simp_idx[i + 1] + 1],
                )
                <= epsilon
                for i in range(len(simp_idx) - 1)
            ]
        )
        assert fd(P, P[simp_idx]) <= epsilon

    check(P_vert, simplify_polyline(P_vert, 2)[1])
    check(P_vert, simplify_polyline(P_vert, 3)[1])
    check(P_vert, simplify_polyline(P_vert, 4)[1])
    check(P_vert, simplify_polyline(P_vert, 5)[1])
    check(P_vert, simplify_polyline(P_vert, 6)[1])
    check(P_vert, simplify_polyline(P_vert, 7)[1])
