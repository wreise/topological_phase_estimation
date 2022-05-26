# Author(s): Wojciech Reise
#
# Copyright (C) 2021 Inria

from gudhi.simplex_tree import SimplexTree
import numpy as np
import sklearn.metrics as sm


def signal_to_simplex_tree(x, periodic=True):
    """Take a one-dimensional signal and create a simplex tree.
    Params:
        x: ndarray (n,) the signal
        periodic: bool, default: True. Add the edge [x[0], x[-1]]
            to filter the circle, instead of the interval
    Returns:
        stree: gudhi.SimplexTree, with simplices of dimension 0 and 1.
    """
    values, indices_in_order = x, np.arange(len(x))
    values = values[np.argsort(indices_in_order)]
    indices = np.arange(len(values))
    edges = [[a, b] for a, b in zip(indices[:-1], indices[1:])]
    if periodic:
        edges.append([indices[-1], indices[0]])  # insert for periodic
    edge_values = list(map(lambda x: max(values[x[0]], values[x[1]]), edges))

    stree = SimplexTree()
    for edge, value in zip(edges, edge_values):
        stree.insert(simplex=edge, filtration=value)

    for index, value in zip(indices, values):
        stree.assign_filtration(simplex=[index], filtration=value)

    return stree

def pairwise_distance(points):
    """Compute the pairwise distance between points."""
    return sm.pairwise_distances(points, metric="chebyshev")


def persistence_distance(points):
    """Difference in persistence of points"""
    persistences = get_persistence(points)
    x = np.abs(persistences[:, None] - persistences[None, :])
    return x


def signal_to_diagram(signal=None, simplex_tree=None):
    """Compute the diagram (in dim 0), with the essential point being assigned the
    maximum value."""
    if simplex_tree is None:
        signal_max = np.max(signal)
        simplex_tree = signal_to_simplex_tree(signal, periodic=True)
    else:
        # signal is None
        signal_max = max([f for _, f in simplex_tree.get_filtration()])
    dgm = simplex_tree.persistence()
    dgm = convert_gudhi_gtda(dgm)
    dgm[np.where(np.isinf(dgm))] = signal_max
    return dgm[:, 0:2]


def convert_gudhi_gtda(pd):
    """Converts a persistence diagram in the form of a list of points
    in (dim, (b,d)) to an array, usable by gtda (array [b,d, dim]).
    It is assumed that it holds a single dimension.
    """
    return np.array([[b, d, dim] for dim, (b, d) in pd])


def get_persistence(points):
    """Return the persistence of points from the diagram.
    Params:
        points: ndarray of shape (n_points, 1), in the birth-death plane.
    Returns:
        ndarray of shape (n_points, 1)
    """
    return np.diff(points, axis=1)


def filter_out(diagram, tau):
    """Filter out points with persistence les than `tau`."""
    persistent_enough = np.diff(diagram, axis=1)[:, 0] > tau
    diagram = diagram[persistent_enough, :]
    return diagram
