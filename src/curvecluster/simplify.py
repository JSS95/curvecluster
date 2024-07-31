"""Polyline simplification."""

import numpy as np
from curvesimilarities.frechet import decision_problem, fd

__all__ = [
    "simplify_polyline",
]


EPSILON = np.finfo(np.float64).eps


def simplify_polyline(P, ell):
    """Vertex-restricted simplification of a polygonal curve."""
    if ell < 2:
        raise ValueError("Cannot simplify to complexity < 2.")
    if ell >= len(P):
        return P, 0.0

    thres_low = 0
    thres_high = fd(P, P[[0, -1]])
    simp = P[FS(P, thres_high)]
    _ell = len(simp)
    thres = thres_high

    # Find large enough threshold
    while _ell > ell:
        thres_low = thres_high
        thres_high = thres_high * 2
        simp = P[FS(P, thres_high)]
        _ell = len(simp)
        thres = thres_high

    while _ell != ell:
        thres = (thres_low + thres_high) / 2
        if (thres - thres_low < EPSILON) | (thres_high - thres < EPSILON):
            raise RuntimeError("Cannot simplify to complexity %i." % ell)
        simp = P[FS(P, thres)]
        _ell = len(simp)
        if _ell > ell:
            thres_low = thres
        else:
            thres_high = thres

    return simp, thres


def FS(P, epsilon):
    ij = 0
    ret = np.empty(len(P), dtype=np.int_)
    count = 0

    ret[count] = ij
    count += 1

    n = len(P)
    while ij < n - 1:
        L = 0
        high = min(2 ** (L + 1), n - ij - 1)
        all_ok = False
        while decision_problem(P[[ij, ij + high]], P[ij : ij + high + 1], epsilon):
            if high == n - ij - 1:
                # stop because all remaining vertices can be simplified
                all_ok = True
                break
            L += 1
            high = min(2 ** (L + 1), n - ij - 1)

        if all_ok:
            ret[count] = n - 1
            count += 1
            break

        low = max(1, high // 2)
        while low < high - 1:
            mid = (low + high) // 2
            if decision_problem(P[[ij, ij + mid]], P[ij : ij + mid + 1], epsilon):
                low = mid
            else:
                high = mid
        ij += min(low, n - ij - 1)
        ret[count] = ij
        count += 1
    return ret[:count]
