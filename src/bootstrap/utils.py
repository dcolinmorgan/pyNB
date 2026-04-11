import numpy as np
import numpy.typing as npt

NDArrayFloat = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]


class NetworkUtils:
    """Utility functions for network analysis."""

    @staticmethod
    def matrix_or(
        matrix: np.ndarray,
        dim: int = 1,
    ) -> NDArrayFloat:
        """Compute element-wise OR operation along specified dimension."""
        if matrix.ndim < dim:
            raise ValueError(f"Input has no dimension {dim}")
        result: NDArrayFloat = np.any(matrix, axis=dim - 1).astype(np.float64)
        return result

    @staticmethod
    def matrix_and(
        matrix: np.ndarray,
        dim: int = 1,
    ) -> NDArrayFloat:
        """Compute element-wise AND operation along specified dimension."""
        cleaned = np.nan_to_num(matrix.astype(np.float64), nan=0.0)
        if cleaned.ndim < dim:
            raise ValueError(f"Input has no dimension {dim}")
        result: NDArrayFloat = np.all(cleaned, axis=dim - 1).astype(np.float64)
        return result

    @staticmethod
    def calc_bin_freq(
        matrix: NDArrayFloat,
        init: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Calculate binned frequencies of matrix values."""
        bins = np.arange(init + 1) / init
        counts, bin_edges = np.histogram(matrix.flatten(), bins=bins)
        freq: NDArrayFloat = counts / counts.sum()
        return freq, bin_edges
