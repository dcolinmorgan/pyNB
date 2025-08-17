import numpy as np
from numpy import linalg
from sklearn.linear_model import LassoCV
from typing import Tuple, Optional, Union
from datastruct.Dataset import Dataset


def Lasso(
    dataset: Union[Dataset, 'Data'],
    alpha_range: Optional[np.ndarray] = None,
    cv: int = 5,
    tol: float = 1e-4,
    max_iter: int = 1000
) -> Tuple[np.ndarray, float]:
    """Infer network matrix A using LASSO regression.
    
    This method solves for A in the equation Y = A^-1*P - E using LASSO regression
    to find a sparse solution. It uses cross-validation to select the optimal
    regularization parameter.
    
    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        alpha_range: Array of alpha values to try. If None, uses logspace(-6, 1, 30)
        cv: Number of folds for cross-validation
        tol: Convergence tolerance for LASSO
        max_iter: Maximum number of iterations for LASSO
        
    Returns:
        Tuple containing:
            - np.ndarray: Inferred network matrix A
            - float: Selected alpha value from cross-validation
            
    Raises:
        ValueError: If Y or P is None or if dimensions don't match
    """
    # Handle both Dataset and Data objects
    # Check if this is a Data object (has _dataset_id) vs a Dataset object
    if hasattr(dataset, '_dataset_id'):
        # This is a Data object, extract the underlying dataset
        actual_dataset = dataset.data
    else:
        # This is already a Dataset object
        actual_dataset = dataset
        
    if actual_dataset.Y is None or actual_dataset.P is None:
        raise ValueError("Dataset must contain Y and P matrices")
        
    Y = actual_dataset.Y
    P = actual_dataset.P
    
    if Y.shape[0] != P.shape[0]:
        raise ValueError("Y and P must have same number of rows")
        
    n_genes = Y.shape[0]
    
    # Set up alpha range if not provided
    if alpha_range is None:
        alpha_range = np.logspace(-6, 1, 30)
        
    # Initialize network matrix
    A = np.zeros((n_genes, n_genes))
    
    # For each gene, solve LASSO regression
    for i in range(n_genes):
        # Target variable is the expression of gene i
        y = Y[i, :]
        
        # Features are the perturbations of all genes
        X = P.T
        
        # Fit LASSO with cross-validation
        lasso = LassoCV(
            alphas=alpha_range,
            cv=cv,
            tol=tol,
            max_iter=max_iter,
            random_state=42
        )
        lasso.fit(X, y)
        
        # Store coefficients in network matrix
        A[i, :] = lasso.coef_
        
    # Ensure diagonal elements are 1 (self-regulation)
    np.fill_diagonal(A, 1.0)
    
    # Return network matrix and mean alpha value
    return A, np.mean([lasso.alpha_ for _ in range(n_genes)])
