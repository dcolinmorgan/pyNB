import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import subprocess
import yaml
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
from datastruct.Dataset import Dataset

def SCENICPLUS(dataset: Optional[Dataset] = None, 
               work_dir: Optional[str] = None, 
               cisTopic_obj_fname: Optional[str] = None, 
               scenic_workflow_dir: Optional[str] = None,
               n_cpu: int = 1,
               keep_files: bool = False,
               run_id: str = '1',
               nested_boot: bool = False,
               nest_runs: int = 50,
               boot_runs: int = 50,
               seed: int = 42,
               fdr: float = 0.05,
               var_names: Optional[List[str]] = None,
               _is_inner_run: bool = False,
               use_snakemake: bool = False,
               use_arboreto: bool = False,
               **kwargs: Any) -> Tuple[np.ndarray, Any]:
    """
    SCENIC+-inspired GRN inference for pyNB.
    
    Uses correlation + GBM importance for TF-target inference without requiring
    the full SCENIC+ stack (no arboreto/dask compatibility issues).
    
    Parameters
    ----------
    dataset : Dataset, optional
        Input dataset containing gene expression data.
    work_dir : str, optional
        Directory to run the analysis in. If None, a temporary directory is created.
    cisTopic_obj_fname : str, optional
        Path to cisTopic object (pickle) for TF list. If not provided or file doesn't exist,
        uses heuristic TF detection.
    scenic_workflow_dir : str, optional
        Path to the directory containing the Snakefile and config/config.yaml.
        Only used if use_snakemake=True.
    n_cpu : int, default=1
        Number of cores to use.
    keep_files : bool, default=False
        Whether to keep the temporary files after execution.
    run_id : str, default='1'
        Run ID for Snakemake (only used if use_snakemake=True).
    nested_boot : bool, default=False
        Whether to run Nested Bootstrap FDR.
    nest_runs : int, default=50
        Number of outer runs (if nested_boot=True).
    boot_runs : int, default=50
        Number of inner runs (if nested_boot=True).
    seed : int, default=42
        Random seed.
    fdr : float, default=0.05
        False Discovery Rate threshold (if nested_boot=True).
    var_names : List[str], optional
        List of gene names to use for the adjacency matrix. If None, inferred from data.
    use_snakemake : bool, default=False
        If True, use Snakemake workflow. If False, use direct Python API (recommended).
    use_arboreto : bool, default=False
        If True, use vendored arboreto for GRN inference (more accurate but requires dask).
        If False, use lightweight correlation+GBM approach (modern dependencies only).
    _is_inner_run : bool, default=False
        Internal flag for nested bootstrap.
        
    Returns
    -------
    adjacency_matrix : numpy.ndarray
        Inferred gene regulatory network (genes x genes).
    eRegulons : list or None
        List of eRegulon objects (only when use_snakemake=False).
    """
    
    # Handle argument shifting if dataset is passed as a string (likely scenic_workflow_dir)
    if isinstance(dataset, str):
        if scenic_workflow_dir is None:
            scenic_workflow_dir = dataset
        dataset = None

    if nested_boot:
        from .nestboot import Nestboot
        nb_config = {'fdr_threshold': fdr}
        nb = Nestboot(nb_config)
        return nb.run_nestboot(
            dataset=dataset,
            inference_method=SCENICPLUS,
            nest_runs=nest_runs,
            boot_runs=boot_runs,
            seed=seed,
            method_kwargs={
                'work_dir': work_dir,
                'cisTopic_obj_fname': cisTopic_obj_fname,
                'scenic_workflow_dir': scenic_workflow_dir,
                'n_cpu': n_cpu,
                'keep_files': keep_files,
                'run_id': run_id,
                'nested_boot': False, # Prevent recursion
                '_is_inner_run': True, # Mark as inner run
                'var_names': var_names,
                **kwargs
            }
        )
    
    # Route to direct Python API or Snakemake
    if not use_snakemake:
        return _run_scenicplus_direct(
            dataset=dataset,
            work_dir=work_dir,
            cisTopic_obj_fname=cisTopic_obj_fname,
            n_cpu=n_cpu,
            keep_files=keep_files,
            seed=seed,
            var_names=var_names,
            use_arboreto=use_arboreto,
            **kwargs
        )
    
    # 0. Setup Workflow Paths (Snakemake mode)
    if scenic_workflow_dir is None:
        scenic_workflow_dir_path = Path(__file__).parent / "scenic_workflow"
    else:
        scenic_workflow_dir_path = Path(scenic_workflow_dir).resolve()
        
    snakefile_path = scenic_workflow_dir_path / "Snakefile"
    config_path = scenic_workflow_dir_path / "config" / "config.yaml"
    
    if not snakefile_path.exists():
        raise FileNotFoundError(f"Snakefile not found at {snakefile_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    # 1. Setup directories
    work_dir_obj = None
    cleanup = False
    
    if work_dir is None:
        if _is_inner_run:
            # For inner runs, use temp dir to avoid conflicts
            work_dir_obj = tempfile.TemporaryDirectory(prefix="scenicplus_run_")
            work_dir = work_dir_obj.name
            cleanup = True
        else:
            # For single run, try to use config output_dir
            try:
                with open(config_path, 'r') as f:
                    tmp_config = yaml.safe_load(f)
                
                out_dir_tmpl = tmp_config.get('params_general', {}).get('output_dir')
                if out_dir_tmpl:
                    # Resolve run_id
                    work_dir = out_dir_tmpl.format(run_id=run_id)
                    # Make absolute if needed (relative to CWD)
                    work_dir = os.path.abspath(work_dir)
                    os.makedirs(work_dir, exist_ok=True)
                    cleanup = False
                else:
                    # Fallback to temp
                    work_dir_obj = tempfile.TemporaryDirectory(prefix="scenicplus_run_")
                    work_dir = work_dir_obj.name
                    cleanup = True
            except Exception as e:
                print(f"Warning: Failed to read config for output_dir: {e}. Using temp dir.")
                work_dir_obj = tempfile.TemporaryDirectory(prefix="scenicplus_run_")
                work_dir = work_dir_obj.name
                cleanup = True
    else:
        os.makedirs(work_dir, exist_ok=True)
        cleanup = False

    # 1.5 Prepare Input Data
    snakemake_config_overrides = [f"run_id={run_id}"]
    
    # Handle Dataset -> AnnData
    if dataset is not None:
        # Extract expression data
        if hasattr(dataset, 'Y') and dataset.Y is not None:
            Y = dataset.Y
        elif hasattr(dataset, 'data') and dataset.data is not None:
            if hasattr(dataset.data, 'Y'):
                Y = dataset.data.Y
            else:
                Y = dataset.data
        else:
            raise ValueError("Dataset provided but could not extract expression matrix Y")
            
        # Extract gene names
        gene_names = None
        if hasattr(dataset, 'names') and dataset.names:
            gene_names = dataset.names
        elif hasattr(dataset, 'data') and hasattr(dataset.data, 'names') and dataset.data.names:
            gene_names = dataset.data.names
        elif hasattr(dataset, '_names') and dataset._names:
             gene_names = dataset._names
             
        if gene_names is None:
            gene_names = [f"Gene_{i}" for i in range(Y.shape[0])]
            
        # Create AnnData (samples x genes)
        # Y is typically (genes x samples) in this codebase
        adata = sc.AnnData(X=Y.T)
        adata.var_names = gene_names
        adata.obs_names = [f"Cell_{i}" for i in range(Y.shape[1])]
        
        # Save to work_dir
        gex_input_path = os.path.join(work_dir, "input_gex.h5ad")
        adata.write_h5ad(gex_input_path)
        
        # Override config
        # Note: Snakemake config overrides via command line are tricky for nested keys.
        # We might need to pass a flat config or use specific syntax.
        # Snakemake supports --config key=value.
        # For nested keys like input_data:GEX_anndata_fname, it's harder.
        # But we can use the fact that the Snakefile uses config["input_data"]["GEX_anndata_fname"]
        # We can try to pass a JSON string or just rely on the fact that we can edit the config file?
        # No, editing config file is bad for concurrency.
        # Better: Create a temporary config file that extends the base one.
        
        # Actually, Snakemake allows overriding nested config via --config input_data={'GEX_anndata_fname': 'path'}
        # But that replaces the whole input_data dict.
        # Let's check how Snakefile uses it.
        
        # Alternative: We can generate a run-specific config file in work_dir
        # and pass that as --configfile.
        
    # Handle cisTopic object
    if cisTopic_obj_fname:
        cisTopic_path = os.path.abspath(cisTopic_obj_fname)
    else:
        cisTopic_path = None

    # Create run-specific config
    with open(config_path, 'r') as f:
        run_config = yaml.safe_load(f)
        
    if dataset is not None:
        run_config['input_data']['GEX_anndata_fname'] = gex_input_path
        
    if cisTopic_path:
        run_config['input_data']['cisTopic_obj_fname'] = cisTopic_path
        
    # Save run config
    run_config_path = os.path.join(work_dir, "config.yaml")
    with open(run_config_path, 'w') as f:
        yaml.dump(run_config, f)

    # We need to run specific rules or 'all'. 
    # The Snakefile produces 'scplus_mdata'.
    # We want 'eRegulons_extended' or 'eRegulons_direct'.
    
    # Use sys.executable -m snakemake to ensure we use the installed package
    cmd = [
        sys.executable, "-m", "snakemake",
        "all",
        "--snakefile", str(snakefile_path),
        "--configfile", str(run_config_path), # Use our new config
        "--cores", str(n_cpu),
        "--config", f"run_id={run_id}"
    ]
    
    # Run
    # Capture output to avoid cluttering stdout unless error
    process = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        raise RuntimeError(f"Snakemake failed with return code {process.returncode}")
    
    # 2. Read Results
    # The output should be in work_dir.
    # Look for eRegulons_extended.tsv
    results_file = os.path.join(work_dir, "eRegulons_extended.tsv")
    if not os.path.exists(results_file):
            # Try direct
            results_file = os.path.join(work_dir, "eRegulons_direct.tsv")
            if not os.path.exists(results_file):
                # Try direct (singular) - as seen in config.yaml
                results_file = os.path.join(work_dir, "eRegulon_direct.tsv")
            
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found at {results_file}. Snakemake output: {process.stdout}")
        
    df = pd.read_csv(results_file, sep='\t')
    
    # 6. Convert to Adjacency Matrix
    # df has columns: TF, Gene, importance, etc.
    # We need to map TF and Gene to indices in var_names.
    
    if var_names is None:
            # Try to infer from config if possible, or just use genes in results
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                gex_path = config.get('input_data', {}).get('GEX_anndata_fname')
                if gex_path and os.path.exists(gex_path):
                    tmp_adata = sc.read_h5ad(gex_path, backed='r')
                    var_names = tmp_adata.var_names.tolist()
            except Exception:
                pass
        
            if var_names is None:
                var_names = sorted(list(set(df['Gene'].unique()) | set(df['TF'].unique())))

    n_genes = len(var_names)
    gene_to_idx = {name: i for i, name in enumerate(var_names)}
    
    adj_matrix = np.zeros((n_genes, n_genes))
    
    for _, row in df.iterrows():
        tf = row['TF']
        target = row['Gene']
        importance = row.get('importance', 1.0) # Or importance_x_rho
        
        if tf in gene_to_idx and target in gene_to_idx:
            i = gene_to_idx[tf]
            j = gene_to_idx[target]
            adj_matrix[i, j] = importance
            
    if cleanup and work_dir_obj:
        work_dir_obj.cleanup()
    elif cleanup and not work_dir_obj:
        shutil.rmtree(work_dir, ignore_errors=True)
            
    return adj_matrix, None


def _run_scenicplus_direct(dataset, work_dir=None, cisTopic_obj_fname=None, n_cpu=1, 
                           keep_files=False, seed=42, var_names=None, use_arboreto=False, **kwargs):
    """Lightweight SCENIC+-inspired implementation with optional arboreto."""
    
    if use_arboreto:
        return _run_with_arboreto(dataset, work_dir, cisTopic_obj_fname, n_cpu, 
                                  keep_files, seed, var_names, **kwargs)
    
    # Lightweight implementation
    import pickle
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Setup work directory
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="scenicplus_")
        cleanup = not keep_files
    else:
        os.makedirs(work_dir, exist_ok=True)
        cleanup = False
    
    # Extract expression data
    if hasattr(dataset, 'Y') and dataset.Y is not None:
        Y = dataset.Y
    elif hasattr(dataset, 'data'):
        Y = dataset.data.Y if hasattr(dataset.data, 'Y') else dataset.data
    else:
        raise ValueError("Cannot extract expression matrix from dataset")
    
    # Extract gene names
    if var_names is None:
        if hasattr(dataset, 'names') and dataset.names:
            var_names = dataset.names
        elif hasattr(dataset, 'data') and hasattr(dataset.data, 'names'):
            var_names = dataset.data.names
        else:
            var_names = [f"Gene_{i}" for i in range(Y.shape[0])]
    
    # Load cisTopic object for TF list
    if cisTopic_obj_fname and os.path.exists(cisTopic_obj_fname):
        with open(cisTopic_obj_fname, 'rb') as f:
            cistopic_obj = pickle.load(f)
        # Extract TF list from cisTopic if available
        tf_list = getattr(cistopic_obj, 'tf_names', None)
    else:
        tf_list = None
    
    # If no TF list, use common TF pattern or all genes
    if tf_list is None:
        tf_list = [g for g in var_names if any(x in g.upper() for x in ['TF', 'FOX', 'SOX', 'HOX', 'ZNF', 'KLF'])]
        if not tf_list:
            tf_list = var_names[:min(100, len(var_names))]  # Use first 100 as potential TFs
    
    # Filter TFs present in data
    tf_indices = [i for i, name in enumerate(var_names) if name in tf_list]
    
    # GRN inference using GBM
    n_genes = len(var_names)
    adj_matrix = np.zeros((n_genes, n_genes))
    
    def infer_targets(tf_idx):
        """Infer targets for a single TF using GBM."""
        X_tf = Y[tf_idx, :].reshape(-1, 1)
        importances = []
        
        for target_idx in range(n_genes):
            if target_idx == tf_idx:
                importances.append(0.0)
                continue
            
            y_target = Y[target_idx, :]
            
            # Quick correlation filter
            corr, _ = spearmanr(X_tf.ravel(), y_target)
            if abs(corr) < 0.1:
                importances.append(0.0)
                continue
            
            # GBM importance
            try:
                gbm = GradientBoostingRegressor(n_estimators=20, max_depth=3, random_state=seed)
                gbm.fit(X_tf, y_target)
                importances.append(gbm.feature_importances_[0] * abs(corr))
            except:
                importances.append(0.0)
        
        return tf_idx, importances
    
    # Parallel inference
    print(f"Inferring GRN for {len(tf_indices)} TFs...")
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {executor.submit(infer_targets, tf_idx): tf_idx for tf_idx in tf_indices}
        
        for future in as_completed(futures):
            tf_idx, importances = future.result()
            adj_matrix[tf_idx, :] = importances
    
    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)
    
    return adj_matrix, None



def _run_with_arboreto(dataset, work_dir=None, cisTopic_obj_fname=None, n_cpu=1,
                       keep_files=False, seed=42, var_names=None, **kwargs):
    """SCENIC+ implementation using vendored arboreto (requires dask)."""
    try:
        from ._vendor.arboreto.algo import grnboost2
    except ImportError as e:
        raise ImportError(f"Vendored arboreto failed to import: {e}. "
                         "Try use_arboreto=False for lightweight mode.")
    
    import pickle
    
    # Setup work directory
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="scenicplus_")
        cleanup = not keep_files
    else:
        os.makedirs(work_dir, exist_ok=True)
        cleanup = False
    
    # Extract expression data
    if hasattr(dataset, 'Y') and dataset.Y is not None:
        Y = dataset.Y
    elif hasattr(dataset, 'data'):
        Y = dataset.data.Y if hasattr(dataset.data, 'Y') else dataset.data
    else:
        raise ValueError("Cannot extract expression matrix from dataset")
    
    # Extract gene names
    if var_names is None:
        if hasattr(dataset, 'names') and dataset.names:
            var_names = dataset.names
        elif hasattr(dataset, 'data') and hasattr(dataset.data, 'names'):
            var_names = dataset.data.names
        else:
            var_names = [f"Gene_{i}" for i in range(Y.shape[0])]
    
    # Load TF list from cisTopic if available
    tf_list = None
    if cisTopic_obj_fname and os.path.exists(cisTopic_obj_fname):
        with open(cisTopic_obj_fname, 'rb') as f:
            cistopic_obj = pickle.load(f)
        tf_list = getattr(cistopic_obj, 'tf_names', None)
    
    if tf_list is None:
        tf_list = [g for g in var_names if any(x in g.upper() for x in ['TF', 'FOX', 'SOX', 'HOX', 'ZNF', 'KLF'])]
        if not tf_list:
            tf_list = var_names[:min(100, len(var_names))]
    
    # Create expression DataFrame for arboreto
    expr_df = pd.DataFrame(Y.T, columns=var_names)
    
    # Run GRNBoost2
    network_df = grnboost2(
        expression_data=expr_df,
        tf_names=tf_list,
        seed=seed,
        verbose=True
    )
    
    # Convert to adjacency matrix
    n_genes = len(var_names)
    gene_to_idx = {name: i for i, name in enumerate(var_names)}
    adj_matrix = np.zeros((n_genes, n_genes))
    
    for _, row in network_df.iterrows():
        tf = row['TF']
        target = row['target']
        importance = row['importance']
        
        if tf in gene_to_idx and target in gene_to_idx:
            adj_matrix[gene_to_idx[tf], gene_to_idx[target]] = importance
    
    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)
    
    return adj_matrix, network_df
