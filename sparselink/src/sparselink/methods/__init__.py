"""Built-in inference methods."""

from sparselink.methods.lasso import LassoMethod
from sparselink.methods.correlation import PartialCorrelation
from sparselink.methods.lsco import LSCOMethod
from sparselink.methods.clr import CLRMethod
from sparselink.methods.elastic_net import ElasticNetMethod, RidgeMethod

__all__ = [
    "LassoMethod",
    "PartialCorrelation",
    "LSCOMethod",
    "CLRMethod",
    "ElasticNetMethod",
    "RidgeMethod",
]
