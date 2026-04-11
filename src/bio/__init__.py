"""pyGS.bio - Biology-specific methods for gene regulatory network inference.

Submodules:
    preprocessing - Expression matrix handling, TF/target gene lists
    wrappers - External bio tool wrappers (scenicplus, pyscenic)
    visualization - Network plots with gene annotations
    evaluation - GRN-specific evaluation (gold standard comparison)
"""

from .preprocessing import load_expression_matrix, filter_tf_targets
from .evaluation import compare_to_gold_standard
from .wrappers import scenicplus_infer, pyscenic_infer

__all__ = [
    "load_expression_matrix",
    "filter_tf_targets",
    "compare_to_gold_standard",
    "scenicplus_infer",
    "pyscenic_infer",
]
