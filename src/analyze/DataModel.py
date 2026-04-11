import numpy as np
from typing import Any, Optional


class DataModel:
    """Base class for analysis utilities."""

    def __init__(self, data: Any = None) -> None:
        self._data: Any = data
        self._tol: float = float(np.finfo(float).eps)
        self._alpha: float = 0.01
        self._type: str = "directed"

    @staticmethod
    def alpha() -> float:
        """Significance level."""
        return 0.01

    @staticmethod
    def type() -> str:
        """Network type."""
        return "directed"

    @staticmethod
    def tol() -> float:
        """Numerical tolerance."""
        return float(np.finfo(float).eps)

    @property
    def data(self) -> Any:
        return self._data
