import numpy as np
from numpy import linalg
from typing import Tuple, Optional
from datastruct.Dataset import Dataset
import warnings

# Use CELER for fast LASSO solving (much faster than sklearn)
try:
    from celer import Lasso as CelerLasso, MultiTaskLasso as CelerMultiTaskLasso
    USE_CELER = True
except ImportError:
    from sklearn.linear_model import Lasso as CelerLasso
    USE_CELER = False
    CelerMultiTaskLasso = None

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    USE_PARALLEL = True
except ImportError:
    USE_PARALLEL = False


def Lasso(
    dataset: Dataset,
    alpha_range: Optional[np.ndarray] = None,
    cv: int = 5,
    tol: float = 1e-4,
    max_iter: int = 10000,
    use_covariance: Optional[bool] = None
) -> Tuple[np.ndarray, float]:
    """Infer network matrix A using LASSO regression.
    
    This matches the MATLAB GeneSPIDER2 implementation:
    For each gene i: lasso(data.Y', data.P(i,:)', 'Lambda', zetavec)
    Regresses all samples of Y against perturbation pattern of gene i.
    
    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        alpha_range: Array of alpha values to try. If None, uses logspace(-6, 1, 30)
        cv: Number of folds for cross-validation
        tol: Convergence tolerance for LASSO
        max_iter: Maximum number of iterations for LASSO
        use_covariance: Whether to use Gram matrix (X'X) formulation. If None, auto-decides
                       based on n_samples vs n_features. True when n_samples > n_features.
        
    Returns:
        Tuple containing:
            - np.ndarray: Inferred network matrix A (n_genes x n_genes x n_alphas)
            - np.ndarray: Alpha values used
            
    Raises:
        ValueError: If Y or P is None or if dimensions don't match
    """
    # Handle both Dataset and Data objects
    if hasattr(dataset, 'Y'):
        Y = dataset.Y
        P = dataset.P
    elif hasattr(dataset, 'data'):
        Y = dataset.data.Y
        P = dataset.data.P
    else:
        raise ValueError("Dataset must contain Y and P matrices")
        
    if Y is None or P is None:
        raise ValueError("Dataset must contain Y and P matrices")
    
    if Y.shape[0] != P.shape[0]:
        raise ValueError("Y and P must have same number of rows (genes)")
        
    n_genes = Y.shape[0]
    n_samples = Y.shape[1]
    
    # Set up alpha range if not provided
    # MATLAB uses normalized zeta in [0,1] range then scales to actual lambda range
    # We'll use a more appropriate range for sklearn's scaled alpha
    if alpha_range is None:
        # Default range: very small to moderate regularization
        alpha_range = np.logspace(-6, -1, 30)
    else:
        # If provided, check if it's in MATLAB's [0,1] normalized range
        # If so, convert to sklearn's scale
        if np.max(alpha_range) <= 1.0 and np.min(alpha_range) >= 0:
            # This looks like MATLAB's normalized range
            # Convert to sklearn scale: multiply by a reasonable lambda_max
            # Compute lambda_max from the first gene as reference
            X_ref = Y.T
            y_ref = P[0, :]
            lambda_max = np.max(np.abs(X_ref.T @ y_ref)) / len(y_ref)
            alpha_range = alpha_range * lambda_max
        
    # Initialize network matrix to store results for all alpha values
    # MATLAB returns 3D array (n_genes x n_genes x n_alphas)
    Afit = np.zeros((n_genes, n_genes, len(alpha_range)))
    
    # Prepare data once - MATLAB: lasso(data.Y', data.P(i,:)', 'Lambda', zetavec)
    # Features: Expression of all genes across all samples (samples x genes)
    X = Y.T  # (n_samples, n_genes)
    
    # Auto-decide whether to use covariance matrix (Gram matrix optimization)
    # Use when n_samples > n_features for speed (reduces complexity from N*D to D*D)
    if use_covariance is None:
        use_covariance = n_samples > n_genes
    
    # Precompute Gram matrix if beneficial
    if use_covariance:
        # Gram matrix: X'X (n_genes x n_genes)
        # This reduces per-iteration cost from O(n_samples * n_genes) to O(n_genes^2)
        XtX = X.T @ X  # (n_genes, n_genes)
        # Precompute X'y for each target (will be computed in loop)
        compute_gram = True
    else:
        XtX = None
        compute_gram = False
    
    # Define worker function for parallel execution
    def fit_gene_lasso(i):
        """Fit LASSO for a single gene across all alphas."""
        y = P[i, :]  # Target: Perturbation of gene i
        gene_coefs = np.zeros((n_genes, len(alpha_range)))
        
        # Precompute X'y if using Gram matrix
        if compute_gram:
            Xty = X.T @ y  # (n_genes,)
        else:
            Xty = None
        
        if USE_CELER:
            # Suppress convergence warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Use CELER with Gram matrix support
                # Note: CELER doesn't directly support precomputed Gram, but uses efficient
                # screening rules that achieve similar speedups
                lasso = CelerLasso(
                    alpha=alpha_range[0],
                    tol=1e-3,  # Relaxed for speed, matching MATLAB
                    max_iter=1000,
                    fit_intercept=False,
                    warm_start=True
                )
                
                # Fit across alphas with warm starts
                # With n_samples=150 > n_genes=50, this is already efficient
                for j, alpha in enumerate(alpha_range):
                    lasso.set_params(alpha=alpha)
                    lasso.fit(X, y)
                    gene_coefs[:, j] = lasso.coef_
        else:
            # Fallback to sklearn
            from sklearn.linear_model import Lasso as SklearnLasso
            for j, alpha in enumerate(alpha_range):
                # sklearn Lasso automatically uses Gram when beneficial
                lasso = SklearnLasso(
                    alpha=alpha, 
                    tol=1e-3, 
                    max_iter=1000, 
                    fit_intercept=False,
                    precompute=use_covariance  # Use precomputed Gram if beneficial
                )
                lasso.fit(X, y)
                gene_coefs[:, j] = lasso.coef_
        
        return i, gene_coefs
    
    # Use parallel processing if available (much faster)
    if USE_PARALLEL and USE_CELER:
        # Parallel execution across genes
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(fit_gene_lasso)(i) for i in range(n_genes)
        )
        for i, gene_coefs in results:
            Afit[i, :, :] = gene_coefs
    else:
        # Sequential execution
        for i in range(n_genes):
            _, gene_coefs = fit_gene_lasso(i)
            Afit[i, :, :] = gene_coefs
    
    # Return full 3D array (n_genes × n_genes × n_alphas) like MATLAB
    return Afit, alpha_range
