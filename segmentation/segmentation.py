# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

import numpy as np

from .estimation_n import estimate_N, estimate_N_with_clustering, estimate_N_with_boosting
from .homology import signal_to_diagram, signal_to_simplex_tree


def _segment_signal(s, tau, estimation_fct):
    """Segment [0,1].
    Params:
        s: 1d array, the input signal
        tau: positive float, the scale parameter tau
        estimation_fct:
    Returns:
        ndarray of shape (K, N_hat), so that each column defines a single segmentation.
    """
    stree = signal_to_simplex_tree(s, periodic=True)
    dgm = signal_to_diagram(simplex_tree=stree)

    # Get birth simplices, more persistent than 2*tau
    persistence_pairs = [(a, b) for a, b in stree.persistence_pairs() if len(b) > 0]
    values_for_pairs = np.array([[stree.filtration(a), stree.filtration(b)]
                                 for a, b in persistence_pairs])
    persistent_pairs = np.diff(values_for_pairs, axis=1)[:, 0] >= tau

    birth_coordinate = np.array(list(map(lambda x: x[0][0], persistence_pairs)))
    persistent_birth_coordinate = birth_coordinate[persistent_pairs]
    persistent_birth_coordinate = np.sort(np.append(persistent_birth_coordinate, [np.argmin(s)]))

    N_hat = estimation_fct(dgm, tau)
    take_every = int(persistent_birth_coordinate.shape[0]/N_hat)
    if N_hat == 1:
        return persistent_birth_coordinate[:, None]
    return persistent_birth_coordinate.reshape(N_hat, take_every).T


def segment_signal(s, tau):
    return _segment_signal(s, tau, estimate_N)


def segment_signal_with_clustering(s, tau):
    return _segment_signal(s, tau, estimate_N_with_clustering)


def segment_with_boosting(s):
    dgm = signal_to_diagram(s)
    N_hat, taus = estimate_N_with_boosting(dgm, estimate_N_with_clustering,
                                           return_mode_taus=True)
    if len(taus) == 0:
        return np.array([[]])
    median_tau = np.median(taus)
    return segment_signal_with_clustering(s, median_tau)
