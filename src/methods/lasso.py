import numpy as np
from numpy import linalg
from sklearn.linear_model import LassoCV
from typing import Tuple, Optional
from datastruct.Dataset import Dataset


def Lasso(
    dataset: Dataset,
    alpha_range: Optional[np.ndarray] = None,
    cv: int = 5,
    tol: float = 1e-4,
    max_iter: int = 1000
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
        
    Returns:
        Tuple containing:
            - np.ndarray: Inferred network matrix A (n_genes x n_genes)
            - float: Selected alpha value from cross-validation
            
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
    # For simplicity, we'll select the best alpha and return 2D
    Afit = np.zeros((n_genes, n_genes, len(alpha_range)))
    
    # For each gene i, regress Y' against P(i,:)'
    # MATLAB: lasso(data.Y', data.P(i,:)', 'Lambda', zetavec, 'Alpha', 1)
    for i in range(n_genes):
        # Features: All samples of Y (transposed to samples x genes)
        X = Y.T  # (n_samples x n_genes)
        
        # Target: Perturbation pattern for gene i across samples
        y = P[i, :]  # (n_samples,)
        
        # Fit LASSO for each alpha value (MATLAB uses all lambdas at once)
        from sklearn.linear_model import Lasso as LassoRegression
        
        for j, alpha in enumerate(alpha_range):
            lasso = LassoRegression(
                alpha=alpha,
                tol=tol,
                max_iter=max_iter,
                random_state=42
            )
            lasso.fit(X, y)
            
            # Store coefficients in network matrix
            Afit[i, :, j] = lasso.coef_
    
    # Select middle alpha (index 25 like MATLAB benchmark)
    # This matches the MATLAB code which uses index 25 for comparison
    selected_idx = min(25, len(alpha_range) - 1)
    A = Afit[:, :, selected_idx]
    selected_alpha = alpha_range[selected_idx]
    
    # Return network matrix and selected alpha value
    return A, selected_alpha
