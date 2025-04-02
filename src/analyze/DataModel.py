import numpy as np

class DataModel:
    """Base class for analysis utilities."""
    
    def __init__(self, data=None):
        self._data = data
        self._tol = np.finfo(float).eps
        self._alpha = 0.01
        self._type = 'directed'

    @staticmethod
    def alpha():
        """Significance level."""
        return 0.01

    @staticmethod
    def type():
        """Network type."""
        return 'directed'

    @staticmethod
    def tol():
        """Numerical tolerance."""
        return np.finfo(float).eps

    @property
    def data(self):
        return self._data
