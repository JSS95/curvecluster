"""Polyline simplification to acquire the initial center."""

import numpy as np

from ._frechet import decision_problem, fd

__all__ = [
    "simplify_polyline",
]


EPSILON = np.finfo(np.float64).eps


def simplify_polyline(P, ell):
    r"""Vertex-restricted simplification of a polygonal curve.

    Let :math:`P` be a polygonal curve with vertices :math:`\{p_0, p_1, ..., p_n\}`.
    The vertex-restricted simplification of :math:`P` with complexity :math:`\ell \le n`
    is a polygonal curve with vertices :math:`\{p_{i_1}, p_{i_2}, ..., p_{i_\ell}\}`
    where :math:`1 = i_1 < \ldots < i_\ell = n`.

    Parameters
    ----------
    P : ndarray
        Polyline to be simplified, as an :math:`n` by :math:`d` array of :math:`n`
        vertices in a :math:`d`-dimensional space.
    ell : int
        Complexity :math:`\ell` of the simplified cuve.

    Returns
    -------
    ndarray
        Simplified polyline, as an :math:`\ell` by :math:`d` array of :math:`\ell`
        vertices in a :math:`d`-dimensional space.

    Raises
    ------
    RuntimeError
        :math:`P` cannot be simplified to exactly :math:`\ell` vertices.
        This is usually due to multiple vertices having same simplification costs.

    Notes
    -----
    This function implements Agarwal et al.'s algorithm [#1]_ as described by Brankovic
    et al. [#2]_.

    References
    ----------
    .. [#1] Agarwal, P. K., Har-Peled, S., et al. (2005). Near-linear time approximation
       algorithms for curve simplification. Algorithmica, 42, 203-219.
    .. [#2] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.
    """
    if ell < 2:
        raise ValueError("Cannot simplify to complexity < 2.")
    if ell >= len(P):
        return P, 0.0

    # Initial threshold value
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
