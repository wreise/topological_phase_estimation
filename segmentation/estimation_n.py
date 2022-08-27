# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

import warnings

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

from .homology import filter_out, pairwise_distance


def estimate_N(diagram, tau, filter_diagonal=True):
    """Estimate N from the diagram.
    Params:
        diagram: shape (*, 2)
        tau: positive float.
    Returns:
         int, the greatest common divisor.
    """
    if filter_diagonal:
        diagram = filter_out(diagram, tau)
    if diagram.shape[0] == 0:
        return 0
    distances = pairwise_distance(diagram)
    close = distances < tau
    close_by_point = np.sum(close.astype(int), axis=1, keepdims=False)
    return np.gcd.reduce(close_by_point)


## Tree
def get_distances_compressed(dgm):
    """Get distances between points and the diagonal. The first dgm.shape[0] values
    correspond to persistence values (twice the distance to the diagonal).
    """
    distance_matrix = pdist(dgm, metric="chebyshev")
    distance_to_diag = np.diff(dgm, axis=1)[:, 0]/2.
    distance_matrix_compressed = np.concatenate([distance_to_diag, distance_matrix])
    return distance_matrix_compressed


def N_estimate_on_tree(dgm, get_gcd=np.gcd.reduce):
    """Compute the function $\tau\mapsto \hat{N}_c(S,\tau)$.
    Params:
        dgm: ndarray of shape (n_points, 2)
    Returns:
        a collection of values for N, at given thresholds.
    """
    n_dgm = dgm.shape[0]
    diagonal_index = 0
    n_points = n_dgm + 1

    distance_matrix_compressed = get_distances_compressed(dgm)
    Z = linkage(distance_matrix_compressed, optimal_ordering=True)
    clusters_merged, thresholds, n_observations = Z[:, [0, 1]], Z[:, 2], Z[:, 3]

    counts_in_non_trivial_clusters = {n: 1 for n in range(1, n_points)}
    Ns = []
    for ind_new, (cluster_a, cluster_b) in enumerate(clusters_merged):
        if (cluster_a == diagonal_index) or (cluster_b == diagonal_index):
            # If new points are added to the diagonal
            diagonal_index = ind_new + n_points
        counts_A, counts_B = [counts_in_non_trivial_clusters.pop(c, 1) for c in [cluster_a, cluster_b]]
        counts_in_non_trivial_clusters.update({ind_new + n_points: counts_A + counts_B})
        all_counts = [
            cluster_size for cluster_index, cluster_size in counts_in_non_trivial_clusters.items()
            if cluster_index != diagonal_index
        ]  # keep all multiplicities except that of the diagonal.
        assert counts_A + counts_B == n_observations[ind_new]

        if len(all_counts) > 0:
            g = get_gcd(all_counts)
        else:
            g = 1
        Ns.append(g)
    return np.array(Ns), thresholds


def N_on_intervals(Ns, thresholds):
    """Convert the output of N_estimate_with_tree (N, thresholds), to a dictionary,
     with N as key, and values being a list of intervals on which h_S is equal to
     the key.
     Params:
        Ns, thresholds: 1d arrays of the same length, representing the function h_S.
    Returns:
        dict of the form {N: [[a_N^1, b_N^1], [a_N^2, b_N^2]], ...}
    """
    n_Ns = Ns.shape[0]
    Ns_dict = {}
    current_index = 0 # init
    while current_index < n_Ns:
        current_N = Ns[current_index]
        further_equal_to_current = Ns[current_index:] == current_N
        different = np.where(np.cumprod(further_equal_to_current) != 1)[0]
        if len(different) > 0:
            first_different = different[0]
            b = thresholds[current_index + first_different]
        else:
            first_different = n_Ns - current_index
            b = np.inf

        interval = [thresholds[current_index], b]
        current_list = Ns_dict.get(current_N, [])
        current_list.append(interval)
        Ns_dict.update({current_N: current_list})
        current_index += first_different
    return Ns_dict


def largest_length(intervals):
    return max([interval[1]-interval[0] for interval in intervals])


def sum_of_lengths(intervals):
    return sum([interval[1]-interval[0] for interval in intervals])


def N_statistics(Ns_dict, map_=largest_length):
    Ns_mapped = {N: map_(intervals) for N, intervals in Ns_dict.items()}
    return Ns_mapped


def get_N(Ns_mapped):
    """Get the maximal value from N, excluding 1.
    Params:
        Ns_mapped: dict {N: f(N)}, for some function f.
    Returns:
        The key with the maximal value, excluding 1.
    """
    Ns_filtered = {N: v for N, v in Ns_mapped.items() if N != 1}
    if len(Ns_filtered) == 0:
        return 1
    return max(Ns_filtered, key=Ns_filtered.get)


def estimate_N_from_tree(dgm, return_intervals=False, get_gcd=np.gcd.reduce):
    """Calculate $\hat{N}_c^T$ on the given diagram. If `return_intervals`, the function
    also returns the interval representation of h_S.
    """
    Ns, thresholds = N_estimate_on_tree(dgm, get_gcd=get_gcd)
    Ns_interval = N_on_intervals(Ns, thresholds)

    N_stat = N_statistics(Ns_interval)
    N = get_N(N_stat)

    if return_intervals:
        return N, Ns_interval
    else:
        return N


def tau_to_N_function(dgm, get_gcd=np.gcd.reduce):
    """Calculate $\tau\mapsto \hat{N}(S,\tau)$."""
    n_pts = dgm.shape[0]
    indices_row, indices_col = np.triu_indices(n_pts + 1, k=1)
    distances_compressed = get_distances_compressed(dgm)
    ordered = np.argsort(distances_compressed)

    counts = np.ones(n_pts).astype(int)
    is_far_from_diag = np.ones(n_pts).astype(bool)

    Ns_log = np.ones(ordered.shape)

    for ind_element, index in enumerate(ordered):
        if index < n_pts:
            # a point gets to the diagonal
            is_far_from_diag[index] = False
        else:
            # raise some counts by 1
            pt_a, pt_b = indices_row[index] - 1, indices_col[index] - 1
            counts[pt_a] += 1
            counts[pt_b] += 1
        Ns_log[ind_element] = get_gcd(counts[is_far_from_diag])
    Ns_dict = N_on_intervals(Ns_log, distances_compressed[ordered])
    return Ns_dict


def estimate_N_with_clustering(diagram, tau, filter_diagonal=True,
                               return_clusterer=False, get_gcd=np.gcd.reduce):
    """Implementation of $\hat{N}_c(S,\tau)$."""
    if filter_diagonal:
        diagram_ = filter_out(diagram, tau)
        if diagram_.shape[0] <= 2:
            return 0
    else:
        diagram_ = diagram
    clusterer = AgglomerativeClustering(n_clusters=None, linkage="single",
                                        distance_threshold=tau,
                                        affinity="chebyshev",
                                        compute_full_tree=True)
    labels = clusterer.fit_predict(diagram_)
    if not(filter_diagonal):
        not_persistent_points = diagram_[:, 1] - diagram_[:, 0] <= tau
        not_persistent_labels = np.unique(labels[not_persistent_points])
    else:
        not_persistent_labels = np.empty((0,))
    clusterer.not_persistent_labels = not_persistent_labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    counts_of_unique = np.array([
        count for label, count in zip(unique_labels, counts) if not (label in not_persistent_labels)
    ], dtype=int)
    gcd = get_gcd(counts)
    if return_clusterer:
        return gcd, clusterer
    else:
        return gcd


def estimate_N_with_boosting(diagram, estimator=estimate_N, return_mode_taus=False):
    """For a range of parameters $\tau$, calculate $\hat{N}_c(\tau)$. Returns the
    mode of the results.
    """
    values_range = np.max(diagram) - np.min(diagram)
    upper_bound_delta = values_range/2.
    tau_max = upper_bound_delta/3.
    tau_max *= 6./5.
    taus = np.linspace(0., tau_max, 40)
    estimated_Ns = [estimator(diagram, tau) for tau in taus]
    not_one_Ns = np.array([N for N in estimated_Ns if (N != 1) and (N != 0)])
    if not_one_Ns.shape[0] <= 1:
        N_hat = -1
    else:
        N_hat = stats.mode(not_one_Ns)[0][0]
    if return_mode_taus:
        filtered_taus = np.array([tau for tau, N in zip(taus, estimated_Ns) if (N != 1) and (N != 0)])
        mode_taus = filtered_taus[N_hat == not_one_Ns]
        return N_hat, mode_taus
    return N_hat
