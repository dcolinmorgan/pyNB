"""sparselink - Domain-agnostic sparse network inference from tabular data."""

from sparselink.types import AdjacencyMatrix, EdgeList, InferenceResult
from sparselink.base import InferenceMethod
from sparselink.registry import registry, get_method, list_methods

__version__ = "0.1.0"

__all__ = [
    "AdjacencyMatrix",
    "EdgeList",
    "InferenceResult",
    "InferenceMethod",
    "registry",
    "get_method",
    "list_methods",
]
