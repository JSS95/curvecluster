"""Initialize cluster centers."""

import numpy as np

from .simplify import simplify_polyline


def gonzalez(curves, k, ell, dist_func):
    r"""Pick *k* initial cluster centers with complexity *ell* from *curves*.

    Parameters
    ----------
    curves : tuple of ndarray
        Curves that are to be clustered. Each curve :math:`P_i` is a :math:`n_i` by
        :math:`d` array of :math:`n_i` vertices in a :math:`d`-dimensional space.
    k : int
        Number of cluster center curves.
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
    This function implements Gonzalez's algorithm [1]_ adapted by
    Brankovic et al. [2]_.

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
