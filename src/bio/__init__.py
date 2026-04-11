"""pyGS.bio - Biology-specific methods for gene regulatory network inference.

Submodules:
    preprocessing - Expression matrix handling, TF/target gene lists
    wrappers - External bio tool wrappers (scenicplus, pyscenic)
    visualization - Network plots with gene annotations
    evaluation - GRN-specific evaluation (gold standard comparison)
"""

from .evaluation import compare_to_gold_standard
from .preprocessing import filter_tf_targets, load_expression_matrix
from .wrappers import pyscenic_infer, scenicplus_infer

__all__ = [
    "load_expression_matrix",
    "filter_tf_targets",
    "compare_to_gold_standard",
    "scenicplus_infer",
    "pyscenic_infer",
]
