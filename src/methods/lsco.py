import numpy as np
from numpy import linalg
from typing import Tuple, Optional, Union
from datastruct.Dataset import Dataset


def LSCO(
    dataset: Union[Dataset, 'Data'],
    tol: float = 1e-8,
    rcond: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Infer network matrix A using least squares regression.
    
    This method solves for A in the equation Y = A^-1*P - E using ordinary least
    squares regression. It provides a non-sparse solution that minimizes the
    sum of squared residuals.
    
    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        tol: Tolerance for singular values in least squares solver
        rcond: Cut-off ratio for small singular values. If None, uses machine
               precision times max(M, N)
               
    Returns:
        Tuple containing:
            - np.ndarray: Inferred network matrix A
            - float: Mean squared error of the fit
            
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
    
    # Initialize network matrix
    A = np.zeros((n_genes, n_genes))
    mse = 0.0
    
    # For each gene, solve least squares regression
    for i in range(n_genes):
        # Target variable is the expression of gene i
        y = Y[i, :]
        
        # Features are the perturbations of all genes
        X = P.T
        
        # Solve least squares
        coef, residuals, rank, s = linalg.lstsq(X, y, rcond=rcond)
        
        # Store coefficients in network matrix
        A[i, :] = coef
        
        # Accumulate mean squared error
        mse += np.sum(residuals) if len(residuals) > 0 else 0.0
        
    # Ensure diagonal elements are 1 (self-regulation)
    np.fill_diagonal(A, 1.0)
    
    # Calculate mean MSE across all genes
    mean_mse = mse / n_genes
    
    return A, mean_mse
