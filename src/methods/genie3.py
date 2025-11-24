"""
GENIE3 (GEne Network Inference with Ensemble of trees) method.
Based on Huynh-Thu et al. (2010) PLoS ONE.

GENIE3 uses random forests to infer gene regulatory networks.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def GENIE3(dataset, threshold_range=None, n_estimators=100, max_features='sqrt', random_state=42):
    """
    GENIE3 network inference using Random Forest regression.
    
    For each gene, train a random forest to predict its expression from all other genes.
    The importance scores from the random forest indicate regulatory relationships.
    
    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    threshold_range : array-like, optional
        Range of threshold values for sparsification (default: logspace(-6, 0, 30))
    n_estimators : int, default=100
        Number of trees in the random forest
    max_features : str or int, default='sqrt'
        Number of features to consider for best split
    random_state : int, default=42
        Random seed for reproducibility
    
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
    
    # Get expression data
    if hasattr(data, 'Y'):
        Y = data.Y  # genes × samples
    elif hasattr(data, 'data'):
        Y = data.data
    else:
        Y = data
    
    n_genes, n_samples = Y.shape
    
    # Initialize importance matrix
    importance_matrix = np.zeros((n_genes, n_genes))
    
    # For each gene, train a random forest to predict it from all others
    for target_gene in range(n_genes):
        # Get target gene expression
        target = Y[target_gene, :]
        
        # Get predictor genes (all except target)
        predictor_indices = np.arange(n_genes) != target_gene
        predictors = Y[predictor_indices, :].T  # samples × (n_genes - 1)
        
        # Skip if not enough samples
        if n_samples < 3:
            continue
        
        # Train random forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )
        
        rf.fit(predictors, target)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Assign importances to the importance matrix
        importance_matrix[predictor_indices, target_gene] = importances
    
    # Create threshold range if not provided
    if threshold_range is None:
        zeta = np.logspace(-6, 0, 30)
    else:
        zeta = np.asarray(threshold_range)
    
    # Scale threshold range based on importance values
    imp_min = np.min(importance_matrix[importance_matrix > 0]) if np.sum(importance_matrix > 0) > 0 else 0
    imp_max = np.max(importance_matrix)
    
    if imp_max > imp_min:
        threshold_range_scaled = imp_min + zeta * (imp_max - imp_min)
    else:
        threshold_range_scaled = zeta * imp_max
    
    # Apply thresholds to create 3D output
    Afit = np.zeros((n_genes, n_genes, len(threshold_range_scaled)))
    
    for k, threshold in enumerate(threshold_range_scaled):
        Afit[:, :, k] = importance_matrix * (importance_matrix >= threshold)
    
    return Afit, threshold_range_scaled
