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
from typing import Optional, Union, List, Dict, Any
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
               **kwargs: Any) -> np.ndarray:
    """
    SCENIC+ inference wrapper for pyGS.
    
    Parameters
    ----------
    dataset : Dataset, optional
        Input dataset containing gene expression data.
    work_dir : str, optional
        Directory to run the analysis in. If None, a temporary directory is created.
    cisTopic_obj_fname : str, optional
        Path to the cisTopic object (pickle). Required for SCENIC+.
    scenic_workflow_dir : str, optional
        Path to the directory containing the Snakefile and config/config.yaml.
        If None, defaults to the bundled workflow directory.
    n_cpu : int, default=1
        Number of cores to use.
    keep_files : bool, default=False
        Whether to keep the temporary files after execution.
    run_id : str, default='1'
        Run ID for Snakemake.
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
    _is_inner_run : bool, default=False
        Internal flag to indicate if this is an inner run of Nested Bootstrap.
        
    Returns
    -------
    adjacency_matrix : numpy.ndarray
        Inferred gene regulatory network (genes x genes).
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
    
    # 0. Setup Workflow Paths
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
            
    return adj_matrix
        