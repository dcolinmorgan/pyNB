import numpy as np
from numpy import linalg
from typing import Tuple, Optional
from datastruct.Dataset import Dataset


def LSCO(
    dataset: Dataset,
    threshold_range: Optional[np.ndarray] = None,
    tol: float = 1e-8,
    rcond: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Infer network matrix A using least squares regression with thresholding.
    
    This matches the MATLAB GeneSPIDER2 implementation:
    Als = -data.P * pinv(data.Y)
    Then applies thresholding at different levels to create sparse networks.
    
    Args:
        dataset: Dataset object containing Y (expression data) and P (perturbations)
        threshold_range: Array of threshold values (normalized 0-1). If None, returns
                        unthresholded network
        tol: Tolerance for singular values in least squares solver
        rcond: Cut-off ratio for small singular values
               
    Returns:
        Tuple containing:
            - np.ndarray: If threshold_range provided, 3D array (n_genes × n_genes × n_thresholds)
                        Otherwise, 2D unthresholded network
            - np.ndarray or float: If threshold_range provided, array of actual thresholds
                                 Otherwise, mean squared error
            
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
        
    # MATLAB: Als = -data.P * pinv(response(data, net))
    # For the basic case (no net provided), response returns Y
    # So: Als = -P * pinv(Y)
    
    # Compute pseudo-inverse of Y
    Y_pinv = linalg.pinv(Y, rcond=rcond)
    
    # Compute unthresholded network matrix: A = -P * pinv(Y)
    Als = -P @ Y_pinv
    
    # If no threshold range provided, return unthresholded network with MSE
    if threshold_range is None:
        try:
            A_pinv = linalg.pinv(Als, rcond=rcond)
            Y_hat = -A_pinv @ P
            mse = np.mean((Y - Y_hat) ** 2)
        except:
            mse = 0.0
        return Als, mse
    
    # Apply thresholding like MATLAB
    # MATLAB code: Convert normalized threshold to actual values
    nonzero_abs = np.abs(Als[Als != 0])
    if len(nonzero_abs) == 0:
        return np.zeros_like(Als), np.array([0.0])
    
    zeta_min = np.min(nonzero_abs) - np.finfo(float).eps
    zeta_max = np.max(nonzero_abs) + 10 * np.finfo(float).eps
    delta = zeta_max - zeta_min
    
    # Convert normalized threshold_range [0,1] to actual thresholds
    actual_thresholds = threshold_range * delta + zeta_min
    
    # Create 3D array: (n_genes, n_genes, n_thresholds)
    n_genes = Als.shape[0]
    estA_3d = np.zeros((n_genes, n_genes, len(actual_thresholds)))
    
    for i, threshold in enumerate(actual_thresholds):
        # Apply threshold: set elements with abs value <= threshold to zero
        Atmp = Als.copy()
        Atmp[np.abs(Atmp) <= threshold] = 0
        estA_3d[:, :, i] = Atmp
    
    return estA_3d, actual_thresholds
