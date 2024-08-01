"""Frechet distance functions."""

# TODO: remove this package and use curvesimilarities after new version is released.

import numpy as np
from numba import njit

EPSILON = np.finfo(np.float64).eps
NAN = np.float64(np.nan)


@njit(cache=True)
def decision_problem(P, Q, epsilon):
    """Return True if frechet distance between P and Q is smaller than epsilon."""
    if len(P.shape) != 2:
        raise ValueError("P must be a 2-dimensional array.")
    if len(Q.shape) != 2:
        raise ValueError("Q must be a 2-dimensional array.")
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same number of columns.")

    P, Q = P.astype(np.float64), Q.astype(np.float64)
    B, L = _reachable_boundaries_1d(P, Q, epsilon)
    if B[-1, 1] == 1 or L[-1, 1] == 1:
        ret = True
    else:
        ret = False
    return ret


@njit(cache=True)
def _reachable_boundaries_1d(P, Q, eps):
    # Equivalent to _reachable_boundaries, but keep 1d array instead of 2d.
    # Memory efficient, but cannot do backtracking.
    p, q = len(P), len(Q)
    B = np.empty((1, 2), dtype=np.float64)
    L = np.empty((q - 1, 2), dtype=np.float64)

    # Construct leftmost Ls
    prevL0_end = 1
    for j in range(q - 1):
        if prevL0_end == 1:
            start, end = _free_interval(Q[j], Q[j + 1], P[0], eps)
            if start == 0:
                L[j] = [start, end]
            else:
                L[j] = [NAN, NAN]
        else:
            L[j] = [NAN, NAN]
        _, prevL0_end = L[j]

    prevB0_end = 1
    for i in range(p - 1):
        # construct lowermost B
        if prevB0_end == 1:
            start, end = _free_interval(P[i], P[i + 1], Q[0], eps)
            if start == 0:
                B[0] = [start, end]
            else:
                B[0] = [NAN, NAN]
        else:
            B[0] = [NAN, NAN]
        _, prevB0_end = B[0]
        for j in range(q - 1):
            prevL_start, _ = L[j]
            prevB_start, _ = B[0]
            L_start, L_end = _free_interval(Q[j], Q[j + 1], P[i + 1], eps)
            B_start, B_end = _free_interval(P[i], P[i + 1], Q[j + 1], eps)

            if not np.isnan(prevB_start):
                L[j] = [L_start, L_end]
            elif prevL_start <= L_end:
                L[j] = [max(prevL_start, L_start), L_end]
            else:
                L[j] = [NAN, NAN]

            if not np.isnan(prevL_start):
                B[0] = [B_start, B_end]
            elif prevB_start <= B_end:
                B[0] = [max(prevB_start, B_start), B_end]
            else:
                B[0] = [NAN, NAN]

    return B, L


@njit(cache=True)
def _free_interval(A, B, P, eps):
    # resulting interval is always in [0, 1] or is [nan, nan].
    coeff1 = B - A
    coeff2 = A - P
    a = np.dot(coeff1, coeff1)
    c = np.dot(coeff2, coeff2) - eps**2
    if a == 0:  # degenerate case
        if c > 0:
            interval = [NAN, NAN]
        else:
            interval = [np.float64(0), np.float64(1)]
        return interval
    b = 2 * np.dot(coeff1, coeff2)
    Det = b**2 - 4 * a * c
    if Det < 0:
        interval = [NAN, NAN]
    else:
        start = max((-b - Det**0.5) / 2 / a, np.float64(0))
        end = min((-b + Det**0.5) / 2 / a, np.float64(1))
        if start > 1 or end < 0:
            start = end = NAN
        interval = [start, end]
    return interval
