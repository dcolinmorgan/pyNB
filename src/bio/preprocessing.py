"""Biology-specific preprocessing for GRN inference.

Handles expression matrices, TF/target gene lists, and regulon formatting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_expression_matrix(
    data: np.ndarray | pd.DataFrame | str,
    gene_names: list[str] | None = None,
    transpose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Load and normalize an expression matrix for GRN inference.

    Parameters
    ----------
    data : array-like or path
        Expression data as (genes x samples) or path to CSV/h5ad file.
    gene_names : list of str, optional
        Gene names. Inferred from DataFrame columns or generated if None.
    transpose : bool
        If True, input is (samples x genes) and will be transposed.

    Returns
    -------
    matrix : np.ndarray
        Expression matrix (genes x samples).
    names : list of str
        Gene names.
    """
    if isinstance(data, str):
        if data.endswith(".h5ad"):
            import scanpy as sc

            adata = sc.read_h5ad(data)
            matrix = (
                adata.X.toarray()
                if hasattr(adata.X, "toarray")
                else np.asarray(adata.X)
            )
            names = list(adata.var_names)
            # scanpy stores (cells x genes), transpose to (genes x samples)
            return matrix.T, names
        else:
            df = pd.read_csv(data, index_col=0)
            names = list(df.index)
            return df.values, names

    if isinstance(data, pd.DataFrame):
        names = list(data.index) if gene_names is None else gene_names
        matrix = data.values
    else:
        matrix = np.asarray(data)
        n_genes = matrix.shape[1 if transpose else 0]
        names = gene_names if gene_names else [f"G{i + 1}" for i in range(n_genes)]

    if transpose:
        matrix = matrix.T

    return matrix, names


def filter_tf_targets(
    gene_names: list[str],
    tf_list: list[str] | None = None,
    tf_file: str | None = None,
) -> tuple[list[str], list[int]]:
    """Filter gene list to known transcription factors.

    Parameters
    ----------
    gene_names : list of str
        All gene names in the expression matrix.
    tf_list : list of str, optional
        Explicit list of TF names.
    tf_file : str, optional
        Path to file with one TF name per line.

    Returns
    -------
    tf_names : list of str
        TF names found in gene_names.
    tf_indices : list of int
        Indices of TFs in gene_names.
    """
    if tf_list is None and tf_file is not None:
        with open(tf_file) as f:
            tf_list = [line.strip() for line in f if line.strip()]

    if tf_list is None:
        # Heuristic: return all genes as potential regulators
        return list(gene_names), list(range(len(gene_names)))

    gene_set = set(gene_names)
    tf_names = [tf for tf in tf_list if tf in gene_set]
    tf_indices = [gene_names.index(tf) for tf in tf_names]
    return tf_names, tf_indices


def format_regulons(
    adjacency: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Convert adjacency matrix to regulon-style edge list.

    Parameters
    ----------
    adjacency : np.ndarray
        (n_genes x n_genes) weighted adjacency matrix.
    gene_names : list of str
        Gene names corresponding to matrix rows/columns.
    tf_names : list of str
        Subset of gene_names that are TFs.
    threshold : float
        Minimum absolute weight to include an edge.

    Returns
    -------
    pd.DataFrame
        Edge list with columns: TF, target, weight.
    """
    tf_set = set(tf_names)
    edges = []
    for i, g in enumerate(gene_names):
        if g not in tf_set:
            continue
        for j, t in enumerate(gene_names):
            if i == j:
                continue
            w = adjacency[i, j]
            if abs(w) > threshold:
                edges.append({"TF": g, "target": t, "weight": float(w)})
    return pd.DataFrame(edges)
