"""
Context Likelihood of Relatedness (CLR) method for network inference.
Based on Faith et al. (2007) PLoS Biology.

CLR uses mutual information with context-based normalization to infer gene regulatory networks.
"""

import numpy as np
from scipy.stats import zscore
from sklearn.metrics import mutual_info_score


def mutual_information_matrix(data):
    """
    Calculate mutual information matrix between all pairs of genes.
    
    Parameters
    ----------
    data : numpy.ndarray
        Gene expression data (genes × samples)
    
    Returns
    -------
    numpy.ndarray
        Mutual information matrix (genes × genes)
    """
    n_genes, n_samples = data.shape
    mi_matrix = np.zeros((n_genes, n_genes))
    
    # Discretize data for MI calculation (using equal-frequency binning)
    n_bins = min(int(np.sqrt(n_samples)), 10)
    discretized = np.zeros_like(data, dtype=int)
    
    for i in range(n_genes):
        # Rank-based discretization
        ranks = np.argsort(np.argsort(data[i, :]))
        discretized[i, :] = (ranks * n_bins) // n_samples
    
    # Calculate MI for all pairs
    for i in range(n_genes):
        for j in range(i, n_genes):
            if i == j:
                mi_matrix[i, j] = 0
            else:
                mi = mutual_info_score(discretized[i, :], discretized[j, :])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
    
    return mi_matrix


def clr_transform(mi_matrix):
    """
    Apply CLR (Context Likelihood of Relatedness) transformation.
    
    Parameters
    ----------
    mi_matrix : numpy.ndarray
        Mutual information matrix
    
    Returns
    -------
    numpy.ndarray
        CLR-transformed matrix
    """
    n_genes = mi_matrix.shape[0]
    clr_matrix = np.zeros((n_genes, n_genes))
    
    # Calculate z-scores for each row and column
    for i in range(n_genes):
        for j in range(n_genes):
            if i != j:
                # Get MI values for gene i (excluding diagonal)
                mi_i = mi_matrix[i, :]
                mi_i_no_diag = mi_i[np.arange(n_genes) != i]
                
                # Get MI values for gene j (excluding diagonal)
                mi_j = mi_matrix[:, j]
                mi_j_no_diag = mi_j[np.arange(n_genes) != j]
                
                # Calculate z-scores
                if np.std(mi_i_no_diag) > 0:
                    z_i = (mi_matrix[i, j] - np.mean(mi_i_no_diag)) / np.std(mi_i_no_diag)
                else:
                    z_i = 0
                
                if np.std(mi_j_no_diag) > 0:
                    z_j = (mi_matrix[i, j] - np.mean(mi_j_no_diag)) / np.std(mi_j_no_diag)
                else:
                    z_j = 0
                
                # CLR score is sqrt of sum of squared z-scores
                clr_matrix[i, j] = np.sqrt(z_i**2 + z_j**2)
    
    return clr_matrix


def CLR(dataset, threshold_range=None):
    """
    Context Likelihood of Relatedness (CLR) network inference.
    
    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    threshold_range : array-like, optional
        Range of threshold values for sparsification (default: logspace(-6, 0, 30))
    
    Returns
    -------
    Afit : numpy.ndarray
        3D array of inferred networks (n_genes × n_genes × n_thresholds)
    threshold_range : numpy.ndarray
        Array of threshold values used
    """
    # Handle both Dataset and Data objects
    if hasattr(dataset, 'data'):
        data = dataset.data
    else:
        data = dataset
    
    # Get expression data (genes × samples)
    if hasattr(data, 'Y'):
        Y = data.Y
    elif hasattr(data, 'data'):
        Y = data.data
    else:
        Y = data
    
    n_genes = Y.shape[0]
    
    # Calculate mutual information matrix
    mi_matrix = mutual_information_matrix(Y)
    
    # Apply CLR transformation
    clr_matrix = clr_transform(mi_matrix)
    
    # Create threshold range if not provided
    if threshold_range is None:
        zeta = np.logspace(-6, 0, 30)
    else:
        zeta = np.asarray(threshold_range)
    
    # Scale threshold range based on CLR values
    clr_min = np.min(clr_matrix[clr_matrix > 0]) if np.sum(clr_matrix > 0) > 0 else 0
    clr_max = np.max(clr_matrix)
    
    if clr_max > clr_min:
        threshold_range_scaled = clr_min + zeta * (clr_max - clr_min)
    else:
        threshold_range_scaled = zeta * clr_max
    
    # Apply thresholds to create 3D output
    Afit = np.zeros((n_genes, n_genes, len(threshold_range_scaled)))
    
    for k, threshold in enumerate(threshold_range_scaled):
        Afit[:, :, k] = clr_matrix * (clr_matrix >= threshold)
    
    # Make network directed by using asymmetric edge weights
    # (keep the CLR scores which are symmetric, but threshold creates sparsity)
    
    return Afit, threshold_range_scaled
