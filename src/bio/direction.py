"""Edge direction inference from perturbation data."""
import numpy as np


def infer_direction(adj: np.ndarray, P: np.ndarray | None = None, Y: np.ndarray | None = None) -> np.ndarray:
    """Infer causal edge direction from perturbation response asymmetry.

    For edge (i,j): if perturbing i affects j more than perturbing j affects i, direction is i→j.

    Parameters
    ----------
    adj : Undirected adjacency matrix (n x n).
    P : Perturbation matrix (n x experiments, identity-like).
    Y : Pre-computed expression matrix (n x experiments). Used as R directly if provided.

    Returns
    -------
    Directed adjacency matrix (n x n).
    """
    n = adj.shape[0]
    if Y is not None:
        R = Y
    else:
        P = P if P is not None else np.eye(n)
        R = np.linalg.solve(np.eye(n) - adj, P)

    directed = np.zeros_like(adj)
    rows, cols = np.where(np.triu(adj != 0))
    for i, j in zip(rows, cols):
        if abs(R[j, i % R.shape[1]]) >= abs(R[i, j % R.shape[1]]):
            directed[i, j] = adj[i, j]
        else:
            directed[j, i] = adj[i, j]
    return directed
