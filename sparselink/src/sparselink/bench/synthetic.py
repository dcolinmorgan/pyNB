"""Synthetic network and expression data generation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def generate_network(
    n_genes: int,
    topology: str = "random",
    sparsity: float = 0.2,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Generate a synthetic adjacency matrix.

    Args:
        n_genes: Number of nodes.
        topology: 'random' or 'scalefree'.
        sparsity: Edge density (0-1).
        seed: Random seed.

    Returns:
        (n_genes x n_genes) adjacency matrix.
    """
    rng = np.random.default_rng(seed)

    if topology == "scalefree":
        return _scalefree(n_genes, sparsity, rng)
    return _random_network(n_genes, sparsity, rng)


def _random_network(
    n: int, sparsity: float, rng: np.random.Generator
) -> npt.NDArray[np.floating]:
    mask = rng.random((n, n)) < sparsity
    np.fill_diagonal(mask, False)
    weights = rng.standard_normal((n, n))
    A = np.where(mask, weights, 0.0)
    # Make symmetric (undirected)
    A = np.triu(A, 1)
    A = A + A.T
    return np.asarray(A)


def _scalefree(
    n: int, sparsity: float, rng: np.random.Generator
) -> npt.NDArray[np.floating]:
    """Barabási-Albert preferential attachment."""
    m = max(1, int(sparsity * n / 2))
    A = np.zeros((n, n))
    # Start with m+1 fully connected nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            w = rng.standard_normal()
            A[i, j] = w
            A[j, i] = w

    for new_node in range(m + 1, n):
        degrees = np.sum(A[:new_node] != 0, axis=1).astype(float)
        total = degrees.sum()
        if total == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degrees / total
        targets = rng.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            w = rng.standard_normal()
            A[new_node, t] = w
            A[t, new_node] = w
    return A


def generate_expression(
    network: npt.NDArray[np.floating],
    n_samples: int = 100,
    snr: float = 10.0,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Generate synthetic expression data from a network.

    Uses a linear model: Y = (I - A)^{-1} P + noise.

    Args:
        network: (n_genes x n_genes) adjacency matrix.
        n_samples: Number of samples (columns).
        snr: Signal-to-noise ratio.
        seed: Random seed.

    Returns:
        (n_samples x n_genes) expression matrix (samples x features).
    """
    rng = np.random.default_rng(seed)
    n = network.shape[0]

    # Stabilize: scale so spectral radius < 1
    eigvals = np.linalg.eigvals(network)
    rho = np.max(np.abs(eigvals))
    if rho > 0:
        A_stable = network / (rho + 0.1)
    else:
        A_stable = network

    G = np.linalg.inv(np.eye(n) - A_stable)
    P = rng.standard_normal((n, n_samples))
    signal = G @ P

    noise_std = np.std(signal) / snr if snr > 0 else 0.0
    noise = (
        rng.normal(0, noise_std, signal.shape)
        if noise_std > 0
        else np.zeros_like(signal)
    )

    Y = signal + noise
    return Y.T  # Return (samples x features)
