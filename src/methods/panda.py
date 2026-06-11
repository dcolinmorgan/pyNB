"""PANDA network inference — wraps netZooPy."""

from typing import Any

import numpy as np
import pandas as pd

from datastruct.Dataset import Dataset


def PANDA(
    dataset: Dataset | Any,
    threshold_range: np.ndarray | None = None,
    motif_file: str | pd.DataFrame | None = None,
    ppi_file: str | pd.DataFrame | None = None,
    alpha: float = 0.1,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Infer gene regulatory network using PANDA (netZooPy).

    PANDA integrates gene expression, TF motif binding, and PPI data
    via message passing to estimate a bipartite TF-gene network.
    Without motif/PPI priors, returns coexpression-based network.

    Parameters
    ----------
    dataset : Dataset
        Input dataset containing Y (genes x samples) expression data.
    threshold_range : np.ndarray, optional
        Threshold values for sparsity control. If provided, returns 3D array.
    motif_file : str or DataFrame, optional
        TF-gene motif prior (tab-separated: TF, gene, weight).
    ppi_file : str or DataFrame, optional
        PPI data (tab-separated: TF1, TF2, weight).
    alpha : float
        PANDA learning rate (default: 0.1).

    Returns
    -------
    adj : np.ndarray
        Adjacency matrix (n_genes x n_genes) or 3D if threshold_range provided.
    thresholds : np.ndarray or None
    """
    try:
        from netZooPy.panda.panda import Panda
    except ImportError as e:
        raise ImportError(
            "netZooPy required: pip install netzoopy"
        ) from e

    # Extract expression from Dataset
    if hasattr(dataset, "data") and hasattr(dataset.data, "Y"):
        Y = dataset.data.Y
        names = getattr(dataset.data, "names", None) or getattr(dataset, "names", None)
    elif hasattr(dataset, "Y"):
        Y = dataset.Y
        names = getattr(dataset, "names", None)
    else:
        raise ValueError("Cannot extract expression from dataset")

    if names is None:
        names = [f"G{i}" for i in range(Y.shape[0])]

    # Y is genes x samples → DataFrame with genes as index
    expr_df = pd.DataFrame(Y, index=names)

    # Run PANDA
    import io as _io
    import sys
    # Suppress PANDA's verbose output
    _old_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    try:
        panda_obj = Panda(
            expression_file=expr_df,
            motif_file=motif_file,
            ppi_file=ppi_file,
            save_memory=False,
            remove_missing=False,
            modeProcess="union",
            alpha=alpha,
            **kwargs,
        )
    finally:
        sys.stdout = _old_stdout

    # Extract adjacency matrix
    n_genes = len(names)
    adj = np.zeros((n_genes, n_genes))

    if hasattr(panda_obj, "panda_network") and isinstance(panda_obj.panda_network, pd.DataFrame):
        # panda_network is TFs x genes when motif prior is provided
        # or genes x genes for coexpression mode
        net = panda_obj.panda_network
        for i, row_name in enumerate(net.index):
            if row_name in names:
                ri = names.index(row_name)
                for j, col_name in enumerate(net.columns):
                    if col_name in names:
                        ci = names.index(col_name)
                        adj[ri, ci] = net.iloc[i, j]
    elif hasattr(panda_obj, "export_panda_results"):
        results = panda_obj.export_panda_results
        if isinstance(results, pd.DataFrame):
            gene2idx = {g: i for i, g in enumerate(names)}
            for _, row in results.iterrows():
                tf = row.get("tf", row.iloc[0])
                gene = row.get("gene", row.iloc[1])
                force = row.get("force", row.iloc[-1])
                if tf in gene2idx and gene in gene2idx:
                    adj[gene2idx[tf], gene2idx[gene]] = force

    np.fill_diagonal(adj, 0.0)

    if threshold_range is not None:
        from methods.scenicplus import _apply_thresholding
        return _apply_thresholding(adj, threshold_range)

    return adj, None
