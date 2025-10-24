import numpy as np
from numpy.typing import NDArray

__all__ = ["renyi_correlation"]


def renyi_correlation(x: NDArray[np.int32], y: NDArray[np.int32]) -> np.float64:
    x = x.ravel()
    y = y.ravel()
    x_vals = np.unique(x)
    y_vals = np.unique(y)
    if len(x_vals) < 2 or len(y_vals) < 2:
        return np.float64(1.0)

    total = len(x)
    assert total == len(y)

    joint = np.empty((len(x_vals), len(y_vals)), dtype=np.float64)

    for i, x_val in enumerate(x_vals):
        for k, y_val in enumerate(y_vals):
            # count how often x_val and y_val co-occur
            joint[i, k] = np.count_nonzero((x == x_val) & (y == y_val)) / total

    marginal_rows = np.sum(joint, axis=0, keepdims=True)
    marginal_cols = np.sum(joint, axis=1, keepdims=True)
    q_matrix = joint / np.sqrt(marginal_rows) / np.sqrt(marginal_cols)
    # singular value decomposition of Q
    singulars = np.linalg.svd(q_matrix, compute_uv=False)

    # return second-largest singular value
    return singulars[1]
