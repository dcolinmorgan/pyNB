"""Wrappers for external biology tools (scenicplus, pyscenic).

These wrap the existing pyGS scenicplus implementation and provide a cleaner
interface for the pyGS.bio subpackage.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Any


def scenicplus_infer(
    expression: np.ndarray,
    gene_names: List[str],
    tf_list: Optional[List[str]] = None,
    n_cpu: int = 1,
    seed: int = 42,
    threshold_range: Optional[np.ndarray] = None,
    use_arboreto: bool = False,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run SCENIC+-inspired GRN inference.

    Thin wrapper around the existing scenicplus method, accepting raw arrays
    instead of Dataset objects.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (genes x samples).
    gene_names : list of str
        Gene names.
    tf_list : list of str, optional
        Transcription factor names. If None, heuristic detection is used.
    n_cpu : int
        Number of CPUs for parallel inference.
    seed : int
        Random seed.
    threshold_range : np.ndarray, optional
        Threshold values for sparsity control.
    use_arboreto : bool
        Use vendored arboreto/GRNBoost2 if True.

    Returns
    -------
    adjacency : np.ndarray
        Inferred adjacency matrix (2D or 3D if threshold_range given).
    thresholds : np.ndarray or None
        Actual thresholds used, or None.
    """
    from methods.scenicplus import SCENICPLUS
    from datastruct.Dataset import Dataset
    from datastruct.Network import Network

    # Build a minimal Dataset object for the legacy interface
    dataset = Dataset()
    dataset._Y = expression
    dataset._names = gene_names
    dataset._P = np.eye(expression.shape[0])
    dataset._network = Network(np.zeros((expression.shape[0], expression.shape[0])))

    return SCENICPLUS(
        dataset=dataset,
        n_cpu=n_cpu,
        seed=seed,
        var_names=gene_names,
        use_arboreto=use_arboreto,
        threshold_range=threshold_range,
        **kwargs,
    )


def pyscenic_infer(
    expression: np.ndarray,
    gene_names: List[str],
    tf_list: Optional[List[str]] = None,
    n_cpu: int = 1,
    seed: int = 42,
) -> np.ndarray:
    """Run pySCENIC-style GRN inference (GRNBoost2 via arboreto).

    This is a convenience wrapper that forces use_arboreto=True.

    Parameters
    ----------
    expression : np.ndarray
        Expression matrix (genes x samples).
    gene_names : list of str
        Gene names.
    tf_list : list of str, optional
        Transcription factor names.
    n_cpu : int
        Number of CPUs.
    seed : int
        Random seed.

    Returns
    -------
    adjacency : np.ndarray
        Inferred adjacency matrix.
    """
    adj, _ = scenicplus_infer(
        expression=expression,
        gene_names=gene_names,
        tf_list=tf_list,
        n_cpu=n_cpu,
        seed=seed,
        use_arboreto=True,
    )
    return adj
