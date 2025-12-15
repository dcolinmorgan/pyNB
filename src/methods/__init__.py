from .lasso import Lasso
from .lsco import LSCO
from .clr import CLR
from .genie3 import GENIE3
from .tigress import TIGRESS
from .scenicplus import SCENICPLUS

def run(method, dataset, nested_boot=False, nest_runs=50, boot_runs=50, seed=42, fdr=0.05, **kwargs):
    """
    Unified runner for all inference methods.
    
    Args:
        method: The inference function (e.g., Lasso, GENIE3) or its name as string.
        dataset: The input dataset.
        nested_boot: Whether to run Nested Bootstrap FDR.
        nest_runs: Number of outer runs (if nested_boot=True).
        boot_runs: Number of inner runs (if nested_boot=True).
        seed: Random seed.
        fdr: False Discovery Rate threshold (if nested_boot=True).
        **kwargs: Arguments passed directly to the inference method.
    """
    # Resolve string method names to functions
    if isinstance(method, str):
        method_map = {
            'lasso': Lasso,
            'lsco': LSCO,
            'clr': CLR,
            'genie3': GENIE3,
            'tigress': TIGRESS,
            'scenicplus': SCENICPLUS
        }
        if method.lower() not in method_map:
            raise ValueError(f"Unknown method: {method}. Available: {list(method_map.keys())}")
        method = method_map[method.lower()]

    if nested_boot:
        from .nestboot import Nestboot
        # Initialize Nestboot with configuration
        nb_config = {'fdr_threshold': fdr}
        nb = Nestboot(nb_config)
        
        return nb.run_nestboot(
            dataset=dataset,
            inference_method=method,
            nest_runs=nest_runs,
            boot_runs=boot_runs,
            seed=seed,
            method_kwargs=kwargs
        )
    else:
        return method(dataset, **kwargs)

__all__ = ['Lasso', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS', 'SCENICPLUS', 'run']
