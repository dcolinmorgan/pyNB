from typing import List, Tuple, Optional
import numpy as np
import numpy.typing as npt
import logging

NDArrayFloat = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]

class NetworkUtils:
    """Utility functions for network analysis."""

    @staticmethod
    def matrix_or(
        matrix: NDArrayFloat,
        dim: int = 1
    ) -> NDArrayFloat:
        """Compute element-wise OR operation along specified dimension.

        Args:
            matrix: Input matrix
            dim: Dimension along which to perform OR operation

        Returns:
            Matrix after OR operation
        """
        if matrix.ndim < dim:
            raise ValueError(f"Input has no dimension {dim}")
            
        return np.any(matrix, axis=dim-1)

    @staticmethod
    def matrix_and(
        matrix: NDArrayFloat,
        dim: int = 1
    ) -> NDArrayFloat:
        """Compute element-wise AND operation along specified dimension.

        Args:
            matrix: Input matrix
            dim: Dimension along which to perform AND operation

        Returns:
            Matrix after AND operation
        """
        matrix = np.nan_to_num(matrix, 0)
        if matrix.ndim < dim:
            raise ValueError(f"Input has no dimension {dim}")
            
        return np.all(matrix, axis=dim-1)

    @staticmethod
    def calc_bin_freq(
        matrix: NDArrayFloat,
        init: int
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Calculate binned frequencies of matrix values.

        Args:
            matrix: Input matrix
            init: Number of bins

        Returns:
            Tuple of (frequencies, bin_edges)
        """
        bins = np.arange(init + 1) / init
        counts, bin_edges = np.histogram(matrix.flatten(), bins=bins)
        freq = counts / counts.sum()
        return freq, bin_edges 


# Standalone function for easier importing
def calc_bin_freq(matrix: NDArrayFloat, init: int) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Calculate binned frequencies of matrix values.
    
    Convenience function that wraps NetworkUtils.calc_bin_freq.
    
    Args:
        matrix: Input matrix
        init: Number of bins
        
    Returns:
        Tuple of (frequencies, bin_edges)
    """
    return NetworkUtils.calc_bin_freq(matrix, init) 
