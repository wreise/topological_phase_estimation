# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

import numpy as np
from scipy.interpolate import interp1d


def get_GP(t_target, sigma, time_scale, n_samples, eps=1e-10, seed=None):
    """Get realizations of a random Gaussian process with square exponential covariance, on [0,1].
    Params:
        t_target: 1d array, discretization of [0,1] to be sampled over.
        sigma: positive float, the variance of the process.
        time_scale: positive float, defines the time scale of the covariance.
        n_samples: posiitve int, number of iid processes to generate
        eps: positive float, positive parameter to help with the conditioning
    Returns:
        ndarray of shape (t_target.shape[0], n_samples), with the GP sample.
    """
    if seed is not None:
        np.random.seed(seed)
    n = 1000
    t = np.linspace(0., 1., n)
    
    cov = lambda t: (sigma**2)*np.exp(-((t[None, :]-t[:, None])/time_scale)**2/2.)
    cov_mat = cov(t) + eps*np.eye(n)
    L = np.linalg.cholesky(cov_mat)
    
    epsilon = np.random.randn(n, n_samples)
    y = np.dot(L, epsilon)
    y_interp = interp1d(t, y, axis=0)(t_target)
    return y_interp


def get_endpoints(N=None, seed=None):
    """Generate a random segmentation of [0,1], represented by the endpoints.
    The segmentation contains at most 40 intervals, and the minimal duration of one is `t_min`.
    Params:
        N: number of intervals.
    Returns:
        ndarray of shape (N+1), with the endpoints of the segmentation.
    """
    N_max = 40
    t_min = 0.02
    if seed is not None:
        np.random.seed(seed)
    if N is None:
        N = np.random.randint(3, N_max)
    if N > N_max:
        warnings.warn(f"Attempting to set N={N}>{N_max}=N_max, which is limited by t_min.")

    n_cuts = N-1
    cuts = np.sort(np.random.rand(n_cuts)*(1-N*t_min)) + t_min*(1. + np.arange(0, n_cuts))
    endpoints = np.array([0., *cuts, 1.])
    return endpoints


def get_distance(N=None, seed=None, n_discr=10000,):
    """Generate a random `gamma`, from a specified `N`."""
    endpoints, Ns = get_endpoints(N=N, seed=seed), np.arange(N+1)
    distance_interp = interp1d(endpoints, Ns, kind="linear")
    
    t = np.linspace(0., 1., n_discr)
    travelled_distance = distance_interp(t)
    return t, travelled_distance, N