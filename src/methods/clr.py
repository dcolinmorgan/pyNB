"""
Context Likelihood of Relatedness (CLR) method for network inference.
Based on Faith et al. (2007) PLoS Biology.

CLR uses mutual information with context-based normalization to infer gene regulatory networks.
"""

import numpy as np
from scipy.stats import zscore
from sklearn.metrics import mutual_info_score
from typing import Union, Optional, Tuple, Any, List
from datastruct.Dataset import Dataset
from analyze.Data import Data


def mutual_information_matrix(data: np.ndarray) -> np.ndarray:
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
    if n_bins < 2:
        n_bins = 2
        
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


def clr_transform(mi_matrix: np.ndarray) -> np.ndarray:
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
    
    # Mask diagonal for stats calculation
    mask = np.ones((n_genes, n_genes), dtype=bool)
    np.fill_diagonal(mask, 0)
    
    means = np.zeros(n_genes)
    stds = np.zeros(n_genes)
    
    for i in range(n_genes):
        row = mi_matrix[i, mask[i, :]]
        means[i] = np.mean(row)
        stds[i] = np.std(row)
        
    # Avoid division by zero
    stds[stds == 0] = 1.0
    
    # Calculate z-scores
    # Z[i, j] is z-score of MI[i, j] using stats of i
    Z = (mi_matrix - means[:, np.newaxis]) / stds[:, np.newaxis]
    
    # CLR[i, j] = sqrt(Z[i, j]^2 + Z[j, i]^2)
    clr_matrix = np.sqrt(Z**2 + Z.T**2)
    np.fill_diagonal(clr_matrix, 0)
    
    return clr_matrix


def CLR(dataset: Union[Dataset, Data, Any], threshold_range: Optional[Union[np.ndarray, List[float]]] = None) -> Tuple[np.ndarray, np.ndarray]:
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
    if hasattr(dataset, 'Y') and dataset.Y is not None:
        Y = dataset.Y
    elif hasattr(dataset, 'data') and dataset.data is not None:
        data = dataset.data
        if hasattr(data, 'Y') and data.Y is not None:
            Y = data.Y
        elif hasattr(data, 'data'):
            Y = data.data
        else:
            Y = data
    else:
        Y = dataset
    
    if not isinstance(Y, np.ndarray):
        raise ValueError("Could not extract expression matrix Y from dataset")
    
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
    pos_vals = clr_matrix[clr_matrix > 0]
    clr_min = np.min(pos_vals) if pos_vals.size > 0 else 0
    clr_max = np.max(clr_matrix)
    
    if clr_max > clr_min:
        threshold_range_scaled = clr_min + zeta * (clr_max - clr_min)
    else:
        threshold_range_scaled = zeta * clr_max
    
    # Apply thresholds to create 3D output
    # Vectorized thresholding
    Afit = clr_matrix[:, :, np.newaxis] * (clr_matrix[:, :, np.newaxis] >= threshold_range_scaled)
    
    return Afit, threshold_range_scaled
