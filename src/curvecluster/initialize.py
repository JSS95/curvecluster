"""Initialize cluster centers."""

import numpy as np
from kmedoids import fasterpam

from .simplify import simplify_polyline

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
