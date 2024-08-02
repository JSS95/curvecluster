"""Polyline averaging functions."""

__all__ = [
    "frechet_centering",
]


def frechet_centering():
    """Compute center curve using Fréchet centering [1]_.

    Notes
    -----
    This method has been also referred to as "free space averaging" [2]_.

    References
    ----------
    .. [1] Buchin, Kevin, et al. "klcluster: Center-based clustering of trajectories."
       Proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in
       Geographic Information Systems. 2019.
    .. [2] Brankovic, M., et al. "(k, l)-Medians Clustering of Trajectories Using
       Continuous Dynamic Time Warping." Proceedings of the 28th International
       Conference on Advances in Geographic Information Systems. 2020.
    """
