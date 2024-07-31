"""Initialize cluster centers."""


def gonzalez(*curves, k, ell):
    r"""Pick *k* initial cluster centers with complexity *ell* from *curves*.

    Parameters
    ----------
    *curves : tuple of ndarray
        Curves that are to be clustered. Each curve :math:`P_i` is a :math:`n_i` by
        :math:`d` array of :math:`n_i` vertices in a :math:`d`-dimensional space.
    k : int
        Number of cluster center curves.
    ell : int
        Complexity :math:`\ell` of the cluster center curves.

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
    """
