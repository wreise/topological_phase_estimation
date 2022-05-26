import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_diagram(diagram, labels=None):
    """Plot a single persistence diagram with plotly."""
    fig = go.Figure(go.Scatter(x=diagram[:, 0], y=diagram[:, 1], mode="markers",
                               marker_color=labels))
    return fig


def plot_segmentation(signal, landmarks, t=None):
    """Plot the signal with the segmentation.
    Params:
        signal: 1d array
        landmarks ndarray of landmarks
        t: 1d array, x values for the plot
    Returns:
        plotly figure with the plots
    """
    if t is None:
        t = np.arange(signal.shape[0])
    traces = [go.Scatter(x=t, y=signal)]
    traces.extend([
        go.Scatter(x=t[landmark], y=signal[landmark], mode="markers",
                   name=r"$t_{:, " + str(ind) + r"}$")
        for ind, landmark in enumerate(landmarks)
    ])
    fig = go.Figure(traces)
    return fig


def plot_intervals(N_estimation):
    """Plot the function $\tau\mapsto \hat{N}_c(S,\tau)$, from the ouput of `estimate_N_with_tree"""
    list_of_intervals = np.array([interval for intervals_ in N_estimation.values() for interval in intervals_])
    values = np.array([n for n, l in N_estimation.items() for _ in range(len(l))])

    order = np.lexsort(list_of_intervals.T, axis=0)
    y, x = np.repeat(values[order], 2), np.array([l for interval in list_of_intervals[order] for l in interval])
    x = np.nan_to_num(x, posinf=np.max(x[np.isfinite(x)]) * 1.1)
    plt.plot(x, y)
    return x, y
