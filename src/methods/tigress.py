"""
TIGRESS (Trustful Inference of Gene REgulation using Stability Selection) method.
Based on Haury et al. (2012) BMC Systems Biology.

TIGRESS combines LARS (Least Angle Regression) with stability selection.
"""

import numpy as np
from sklearn.linear_model import LassoLarsIC, Lars


def tigress_single_gene(target_expr, predictor_expr, n_bootstrap=100, alpha_range=None, random_state=42):
    """
    Run TIGRESS for a single target gene.
    
    Parameters
    ----------
    target_expr : numpy.ndarray
        Expression of target gene (samples,)
    predictor_expr : numpy.ndarray
        Expression of predictor genes (samples × n_predictors)
    n_bootstrap : int
        Number of bootstrap samples for stability selection
    alpha_range : array-like, optional
        Range of regularization parameters
    random_state : int
        Random seed
    
    Returns
    -------
    scores : numpy.ndarray
        Stability scores for each predictor
    """
    n_samples, n_predictors = predictor_expr.shape
    
    if n_samples < 3 or n_predictors == 0:
        return np.zeros(n_predictors)
    
    # Count how many times each feature is selected across bootstraps
    selection_counts = np.zeros(n_predictors)
    
    np.random.seed(random_state)
    
    for boot_idx in range(n_bootstrap):
        # Bootstrap sample
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = predictor_expr[bootstrap_indices, :]
        y_boot = target_expr[bootstrap_indices]
        
        try:
            # Use LassoLarsIC for automatic alpha selection (BIC criterion)
            model = LassoLarsIC(criterion='bic', max_iter=500)
            model.fit(X_boot, y_boot)
            
            # Count non-zero coefficients
            selected = np.abs(model.coef_) > 1e-10
            selection_counts += selected.astype(float)
            
        except Exception as e:
            # If LassoLarsIC fails, use regular LARS
            try:
                model = Lars(n_nonzero_coefs=min(5, n_predictors), fit_intercept=True)
                model.fit(X_boot, y_boot)
                selected = np.abs(model.coef_) > 1e-10
                selection_counts += selected.astype(float)
            except:
                # If that also fails, skip this bootstrap
                continue
    
    # Stability scores = proportion of bootstrap samples where feature was selected
    stability_scores = selection_counts / n_bootstrap
    
    return stability_scores


def TIGRESS(dataset, threshold_range=None, n_bootstrap=50, random_state=42):
    """
    TIGRESS network inference using stability selection with LARS.
    
    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    threshold_range : array-like, optional
        Range of threshold values for sparsification (default: logspace(-6, 0, 30))
    n_bootstrap : int, default=50
        Number of bootstrap samples for stability selection
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
    
    # Initialize stability score matrix
    stability_matrix = np.zeros((n_genes, n_genes))
    
    # For each gene, run stability selection
    for target_gene in range(n_genes):
        # Get target gene expression
        target = Y[target_gene, :]
        
        # Get predictor genes (all except target)
        predictor_indices = np.arange(n_genes) != target_gene
        predictors = Y[predictor_indices, :].T  # samples × (n_genes - 1)
        
        # Run TIGRESS for this gene
        stability_scores = tigress_single_gene(
            target, 
            predictors, 
            n_bootstrap=n_bootstrap,
            random_state=random_state + target_gene  # Different seed per gene
        )
        
        # Assign stability scores to the matrix
        stability_matrix[predictor_indices, target_gene] = stability_scores
    
    # Create threshold range if not provided
    if threshold_range is None:
        zeta = np.logspace(-6, 0, 30)
    else:
        zeta = np.asarray(threshold_range)
    
    # Scale threshold range based on stability scores (which are in [0, 1])
    # Use the actual range of non-zero scores
    stab_min = np.min(stability_matrix[stability_matrix > 0]) if np.sum(stability_matrix > 0) > 0 else 0
    stab_max = np.max(stability_matrix)
    
    if stab_max > stab_min:
        threshold_range_scaled = stab_min + zeta * (stab_max - stab_min)
    else:
        threshold_range_scaled = zeta * stab_max if stab_max > 0 else zeta * 0.5
    
    # Apply thresholds to create 3D output
    Afit = np.zeros((n_genes, n_genes, len(threshold_range_scaled)))
    
    for k, threshold in enumerate(threshold_range_scaled):
        Afit[:, :, k] = stability_matrix * (stability_matrix >= threshold)
    
    return Afit, threshold_range_scaled


def tigress_base_single_gene(target_expr, predictor_expr, random_state=42):
    """
    Run TIGRESS base learner (LassoLarsIC) for a single target gene without bootstrapping.
    
    Parameters
    ----------
    target_expr : numpy.ndarray
        Expression of target gene (samples,)
    predictor_expr : numpy.ndarray
        Expression of predictor genes (samples × n_predictors)
    random_state : int
        Random seed
    
    Returns
    -------
    coefs : numpy.ndarray
        Coefficients for each predictor
    """
    n_samples, n_predictors = predictor_expr.shape
    
    if n_samples < 3 or n_predictors == 0:
        return np.zeros(n_predictors)
    
    np.random.seed(random_state)
    
    try:
        # Use LassoLarsIC for automatic alpha selection (BIC criterion)
        model = LassoLarsIC(criterion='bic', max_iter=500)
        model.fit(predictor_expr, target_expr)
        coefs = model.coef_
        
    except Exception as e:
        # If LassoLarsIC fails, use regular LARS
        try:
            model = Lars(n_nonzero_coefs=min(5, n_predictors), fit_intercept=True)
            model.fit(predictor_expr, target_expr)
            coefs = model.coef_
        except:
            coefs = np.zeros(n_predictors)
            
    return coefs


def TIGRESS_base(dataset, random_state=42, **kwargs):
    """
    TIGRESS base learner (LassoLarsIC) without stability selection.
    Intended for use within NestBoot which handles the bootstrapping.
    
    Parameters
    ----------
    dataset : Dataset or Data object
        Input dataset containing gene expression data
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    A : numpy.ndarray
        Adjacency matrix (n_genes × n_genes)
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
    
    # Initialize adjacency matrix
    A = np.zeros((n_genes, n_genes))
    
    # For each gene, run regression
    for target_gene in range(n_genes):
        # Get target gene expression
        target = Y[target_gene, :]
        
        # Get predictor genes (all except target)
        predictor_indices = np.arange(n_genes) != target_gene
        predictors = Y[predictor_indices, :].T  # samples × (n_genes - 1)
        
        # Run regression for this gene
        coefs = tigress_base_single_gene(
            target, 
            predictors, 
            random_state=random_state + target_gene
        )
        
        # Assign coefficients to the matrix
        A[predictor_indices, target_gene] = np.abs(coefs)
    
    return A
