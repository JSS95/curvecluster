"""Initialize cluster centers."""

import numpy as np
from curvesimilarities import fd
from kmedoids import fasterpam

from ._frechet import decision_problem

__all__ = [
    "gonzalez",
    "gonzalez_pam",
]


def gonzalez(curves, k, ell, dist_func):
    r"""Initialize curve cluster centers with Gonzalez's algorithm [1]_.

    Parameters
    ----------
    curves : tuple of ndarray
        Curves that are to be clustered. Each curve :math:`P_i` is a :math:`n_i` by
        :math:`d` array of :math:`n_i` vertices in a :math:`d`-dimensional space.
    k : int
        Number of cluster center curves to be initialized.
    ell : int
        Complexity :math:`\ell` of the cluster center curves.
    dist_func : function
        Distance function between curves.

    Returns
    -------
    centers : ndarray
        Cluster centers, as a :math:`k` by :math:`\ell` by :math:`d` array of vertices.

    Notes
    -----
    This function implements Gonzalez's algorithm adapted by Brankovic et al. [2]_.

    References
    ----------
    .. [1] Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster
       distance. Theoretical computer science, 38, 293-306.
    .. [2] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.

    Examples
    --------
    >>> from curvesimilarities import fd
    >>> from numpy.random import normal
    >>> t = np.linspace(0, np.pi, 20)
    >>> Ps = [np.stack([t, np.sin(t) + normal(0, 0.01, len(t))]).T for _ in range(10)]
    >>> Qs = [np.stack([t, np.cos(t) + normal(0, 0.01, len(t))]).T for _ in range(10)]
    >>> centers = gonzalez(Ps + Qs, 2, 10, fd)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> for P in Ps: plt.plot(*P.T, "--", color="gray")  # doctest: +SKIP
    >>> for Q in Qs: plt.plot(*Q.T, "--", color="gray")  # doctest: +SKIP
    >>> plt.plot(*centers.transpose(2, 1, 0))  # doctest: +SKIP
    """
    centers = np.empty((k, ell, curves[0].shape[1]), dtype=np.float64)
    dists = np.empty((k, len(curves)), dtype=np.float64)

    idx_0 = np.random.randint(len(curves))
    c_0, _ = simplify_polyline(curves[idx_0], ell)

    centers[0, :] = c_0
    for idx in range(len(curves)):
        if idx == idx_0:
            d = 0.0
        else:
            d = dist_func(c_0, curves[idx])
        dists[0, idx] = d

    for i in range(1, k):
        idx_i = np.argmax(np.min(dists[:i], axis=0))
        c_i, _ = simplify_polyline(curves[idx_i], ell)

        centers[i, :] = c_i
        for idx in range(len(curves)):
            if idx == idx_i:
                d = 0.0
            else:
                d = dist_func(c_i, curves[idx])
            dists[i, idx] = d

    return centers


def gonzalez_pam(curves, k, ell, dist_func):
    r"""Initialize cluster centers with Gonzalez's algorithm [1]_ and PAM [2]_.

    Parameters
    ----------
    curves : tuple of ndarray
        Curves that are to be clustered. Each curve :math:`P_i` is a :math:`n_i` by
        :math:`d` array of :math:`n_i` vertices in a :math:`d`-dimensional space.
    k : int
        Number of cluster center curves to be initialized.
    ell : int
        Complexity :math:`\ell` of the cluster center curves.
    dist_func : function
        Distance function between curves.

    Returns
    -------
    centers : ndarray
        Cluster centers, as a :math:`k` by :math:`\ell` by :math:`d` array of vertices.

    Notes
    -----
    This function implements the method described by Brankovic et al. [3]_, and uses
    PAM implemented by Schubert and Lenssen [4]_.

    References
    ----------
    .. [1] Gonzalez, T. F. (1985). Clustering to minimize the maximum intercluster
       distance. Theoretical computer science, 38, 293-306.
    .. [2] Kaufmann, Leonard & Rousseeuw, Peter. (1987). Clustering by Means of Medoids.
       Data Analysis based on the L1-Norm and Related Methods. 405-416.
    .. [3] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.
    .. [4] Schubert, E., & Lenssen, L. (2022). Fast k-medoids Clustering in Rust and
       Python. Journal of Open Source Software, 7(75), 4183.

    Examples
    --------
    >>> from curvesimilarities import fd
    >>> from numpy.random import normal
    >>> t = np.linspace(0, np.pi, 20)
    >>> Ps = [np.stack([t, np.sin(t) + normal(0, 0.01, len(t))]).T for _ in range(10)]
    >>> Qs = [np.stack([t, np.cos(t) + normal(0, 0.01, len(t))]).T for _ in range(10)]
    >>> centers = gonzalez_pam(Ps + Qs, 2, 10, fd)
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    >>> for P in Ps: plt.plot(*P.T, "--", color="gray")  # doctest: +SKIP
    >>> for Q in Qs: plt.plot(*Q.T, "--", color="gray")  # doctest: +SKIP
    >>> plt.plot(*centers.transpose(2, 1, 0))  # doctest: +SKIP
    """
    simp_curves = np.empty((len(curves), ell, curves[0].shape[-1]), dtype=np.float64)
    for i in range(len(curves)):
        simp, _ = simplify_polyline(curves[i], ell)
        simp_curves[i] = simp

    dist_mat = np.empty((len(simp_curves), len(curves)), dtype=np.float64)
    for i in range(len(simp_curves)):
        for j in range(len(curves)):
            if i == j:
                d = 0.0
            else:
                d = dist_func(simp_curves[i], curves[j])
            dist_mat[i, j] = d

    # gonzalez
    center_idxs = np.empty(k, dtype=np.int_)
    center_idxs[0] = np.random.randint(len(simp_curves))
    for i in range(1, k):
        center_idxs[i] = np.argmax(np.min(dist_mat[center_idxs[:i]], axis=0))

    # pam
    center_idxs = fasterpam(dist_mat, center_idxs).medoids
    return simp_curves[center_idxs]


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
