"""Built-in inference methods."""

from sparselink.methods.lasso import LassoMethod
from sparselink.methods.correlation import PartialCorrelation

__all__ = ["LassoMethod", "PartialCorrelation"]
