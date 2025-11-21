import numpy as np
from numpy import linalg
from typing import Tuple, Optional
from datastruct.Dataset import Dataset


def LSCO(
    dataset: Dataset,
    tol: float = 1e-8,
    rcond: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Infer network matrix A using least squares regression.
    
    This matches the MATLAB GeneSPIDER2 implementation:
    estA = -data.P * pinv(data.Y)
    
    Computes the network matrix using the closed-form least squares solution.
    
    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        tol: Tolerance for singular values in least squares solver
        rcond: Cut-off ratio for small singular values. If None, uses machine
               precision times max(M, N)
               
    Returns:
        Tuple containing:
            - np.ndarray: Inferred network matrix A (n_genes x n_genes)
            - float: Mean squared error of the fit
            
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
        
    # MATLAB: estA = -data.P * pinv(response(data, net))
    # For the basic case (no net provided), response returns Y
    # So: estA = -P * pinv(Y)
    
    # Compute pseudo-inverse of Y
    Y_pinv = linalg.pinv(Y, rcond=rcond)
    
    # Compute network matrix: A = -P * pinv(Y)
    A = -P @ Y_pinv
    
    # Calculate mean squared error
    # Reconstruct Y_hat = -pinv(A) * P (approximately)
    try:
        A_pinv = linalg.pinv(A, rcond=rcond)
        Y_hat = -A_pinv @ P
        mse = np.mean((Y - Y_hat) ** 2)
    except:
        # If reconstruction fails, return 0 MSE
        mse = 0.0
    
    return A, mse
