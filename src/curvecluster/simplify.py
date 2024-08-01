"""Polyline simplification to acquire the initial centers of cluster."""

import numpy as np
from curvesimilarities import fd

from ._frechet import decision_problem

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
    simp : ndarray
        Simplified polyline, as an :math:`\ell` by :math:`d` array of :math:`\ell`
        vertices in a :math:`d`-dimensional space.
    thres : double
        Threshold value which yields simplified polyline with highest accessible
        complexity :math:`\ell' \le \ell`.

    Notes
    -----
    This function implements Agarwal et al.'s algorithm [1]_ as described by Brankovic
    et al. [2]_, using Fréchet distance.

    In case where binary search does not converge to desired :math:`\ell`, this function
    simplifies :math:`P` to the highest accessible complexity :math:`\ell' < \ell`.
    Then, :math:`\ell - \ell'` vertices are selected from :math:`P` and added the
    simplified polyline.

    References
    ----------
    .. [1] Agarwal, P. K., Har-Peled, S., et al. (2005). Near-linear time
       approximation algorithms for curve simplification. Algorithmica, 42, 203-219.
    .. [2] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.

    Examples
    --------
    >>> t = np.linspace(0, np.pi, 500)
    >>> P = np.stack([t, np.sin(t) + np.random.normal(0, 0.01, len(t))]).T
    >>> P_simp, _ = simplify_polyline(P, 10)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> plt.plot(*P.T); plt.plot(*P_simp.T)  # doctest: +SKIP
    """
    if ell < 2:
        raise ValueError("Cannot simplify to complexity < 2.")
    if ell >= len(P):
        return P, 0.0

    # Initial threshold value
    thres_low = 0
    thres_high = fd(P, P[[0, -1]])
    simp_idx = FS(P, thres_high)
    _ell = len(simp_idx)
    thres = thres_high

    # Find large enough threshold
    while _ell > ell:
        thres_low = thres_high
        thres_high = thres_high * 2
        simp_idx = FS(P, thres_high)
        _ell = len(simp_idx)
        thres = thres_high
    idx_less = simp_idx

    add_vertices = False
    while _ell != ell:
        thres = (thres_low + thres_high) / 2
        if (thres - thres_low < EPSILON) | (thres_high - thres < EPSILON):
            add_vertices = True
            break
        simp_idx = FS(P, thres)
        _ell = len(simp_idx)
        if _ell > ell:
            thres_low = thres
        else:
            thres_high = thres
            idx_less = simp_idx

    if add_vertices:
        _ell = len(idx_less)
        thres = thres_high
        while _ell < ell:
            idx_maxdiff = np.argmax(np.diff(idx_less))
            new_idx = np.array(
                ((idx_less[idx_maxdiff] + idx_less[idx_maxdiff + 1]) // 2,)
            )
            idx_less = np.concatenate(
                (idx_less[: idx_maxdiff + 1], new_idx, idx_less[idx_maxdiff + 1 :])
            )
            _ell = len(idx_less)
        simp_idx = idx_less

    return P[simp_idx], thres


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
