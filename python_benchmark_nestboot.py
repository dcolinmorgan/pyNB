import sys
import os
import json
import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')
from analyze.Data import Data
from datastruct.Network import Network
from methods.lasso import Lasso
from methods.lsco import LSCO
from analyze.CompareModels import CompareModels
from bootstrap.nb_fdr import NetworkBootstrap

def run_comparison_analysis(true_net, inferred_net):
    """Compare inferred network with true network."""
    comp = CompareModels(true_net, Network(inferred_net))
    
    try:
        from sklearn.metrics import roc_auc_score
        true_flat = (true_net.A != 0).astype(float).flatten()
        inferred_flat = np.abs(inferred_net.flatten())
        
        if len(np.unique(true_flat)) > 1 and np.sum(inferred_flat) > 0:
            auroc = roc_auc_score(true_flat, inferred_flat)
        else:
            auroc = 0.5
    except:
        auroc = 0.5
    
    return {
        'f1_score': comp.F1[0] if len(comp.F1) > 0 else 0.0,
        'auroc': auroc,
        'precision': comp.pre[0] if len(comp.pre) > 0 else 0.0,
        'recall': comp.sen[0] if len(comp.sen) > 0 else 0.0
    }

def find_network_file(network_dir, network_id):
    """Find network file recursively with the given ID."""
    # Look for files containing the ID
    matches = list(Path(network_dir).rglob(f"*{network_id}*.json"))
    
    if matches:
        return matches[0]
    
    # Try different patterns if not found
    patterns = [
        f"**/*ID{network_id}.json",
        f"**/*ID{network_id}*",
        f"**/*{network_id}.json",
        f"**/*{network_id}*"
    ]
    
    for pattern in patterns:
        matches = list(Path(network_dir).rglob(pattern))
        if matches:
            return matches[0]
    
    return None

def run_nestboot_method(method_name, data, net, zetavec, n_init, n_boot, fdr, seed=42):
    """Run NestBoot for a specific method."""
    np.random.seed(seed)
    
    # Initialize NetworkBootstrap
    nb = NetworkBootstrap()
    
    start_time = time.time()
    
    # Create method function
    def inference_method(dataset, zetavec=None):
        if method_name == 'LASSO':
            A_3d, _ = Lasso(dataset, alpha_range=zetavec)
            return A_3d
        elif method_name == 'LSCO':
            A_3d, _ = LSCO(dataset, threshold_range=zetavec)
            return A_3d
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    # Run NestBoot
    results = nb.run_nestboot(
        dataset=data,
        inference_method=lambda ds, **kwargs: inference_method(ds, zetavec),
        nest_runs=n_init,
        boot_runs=n_boot,
        seed=seed,
        method_kwargs={}
    )
    
    exec_time = time.time() - start_time
    
    # Reconstruct the adjacency matrix from gene pairs
    binary_network = np.zeros((data.data.N, data.data.N))
    for idx, (gene_i, gene_j) in enumerate(zip(results.gene_i, results.gene_j)):
        # Extract gene indices from names like "Gene_00", "Gene_01", etc.
        i = int(gene_i.split('_')[1])
        j = int(gene_j.split('_')[1])
        binary_network[i, j] = results.xnet[idx]
    
    # Compare with true network
    metrics = run_comparison_analysis(net, binary_network)
    
    # Return results
    result = {
        'method': method_name,
        'n_init': n_init,
        'n_boot': n_boot,
        'fdr': fdr,
        'time': exec_time,
        'auroc': metrics['auroc'],
        'f1': metrics['f1_score'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'density': np.sum(binary_network != 0) / binary_network.size,
        'support_threshold': results.support,
        'fp_rate': results.fp_rate
    }
    
    return result

def main():
    # Configuration - matching MATLAB exactly
    N_INIT = 10      # Outer loop iterations (reduced for faster testing)
    N_BOOT = 10      # Inner loop iterations (reduced for faster testing)
    FDR = 5         # FDR percentage
    OUTPUT_DIR = 'benchmark_results'
    FIGURES_DIR = 'benchmark_figures'
    
    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    figures_dir = Path(FIGURES_DIR)
    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    
    # Dataset paths
    dataset_root = os.path.expanduser('../GeneSPIDER2/data/gs-datasets/N50')
    network_root = os.path.expanduser('../GeneSPIDER2/data/gs-networks')
    
    # Zetavec - CRITICAL: must match MATLAB exactly
    zetavec = np.logspace(-6, 0, 30)
    
    print(f"🚀 Python NestBoot N50 Benchmark (Init={N_INIT}, Boot={N_BOOT}, FDR={FDR}%)")
    print("=" * 70)
    
    # Find datasets
    dataset_files = sorted(glob.glob(os.path.join(dataset_root, "*.json")))
    print(f"Found {len(dataset_files)} N50 datasets.")
    
    # Results storage
    all_results = []
    results_file = output_dir / 'python_nestboot_results.csv'
    
    # Check if file exists to determine if we need headers
    file_exists = results_file.exists()
    
    # Process all datasets
    processed_count = 0
    
    for dataset_path in dataset_files:
        dataset_filename = os.path.basename(dataset_path)
        print(f"\n🔄 [{processed_count+1}/{len(dataset_files)}] Processing {dataset_filename}")
        
        try:
            # Load dataset
            data = Data.from_json_file(dataset_path)
            
            # Extract network ID
            network_id = None
            if hasattr(data.data, 'network') and data.data.network:
                network_id = data.data.network.network.split('-ID')[-1]
            else:
                with open(dataset_path, 'r') as f:
                    json_data = json.load(f)
                if 'obj_data' in json_data and 'network' in json_data['obj_data']:
                    network_id = json_data['obj_data']['network'].split('-ID')[-1]
            
            if not network_id:
                print(f"   ⚠️ Could not extract network ID from {dataset_filename}")
                continue
            
            # Find network file
            network_path = find_network_file(network_root, network_id)
            if not network_path:
                print(f"   ⚠️ Network file not found for ID {network_id}")
                continue
                
            # Load network
            net = Network.from_json_file(str(network_path))
            
            print(f"   📂 Network: {os.path.basename(str(network_path))}")
            
            # Run LASSO NestBoot
            print("   Running LASSO NestBoot...")
            res_lasso = run_nestboot_method('LASSO', data, net, zetavec, N_INIT, N_BOOT, FDR)
            res_lasso['dataset'] = dataset_filename
            res_lasso['network'] = os.path.basename(str(network_path))
            all_results.append(res_lasso)
            # Save after each result (append mode)
            df_result = pd.DataFrame([res_lasso])
            df_result.to_csv(results_file, mode='a', header=not file_exists, index=False)
            file_exists = True  # Headers written after first save
            print(".3f")
            
            # Run LSCO NestBoot
            print("   Running LSCO NestBoot...")
            res_lsco = run_nestboot_method('LSCO', data, net, zetavec, N_INIT, N_BOOT, FDR)
            res_lsco['dataset'] = dataset_filename
            res_lsco['network'] = os.path.basename(str(network_path))
            all_results.append(res_lsco)
            # Save after each result (append mode)
            df_result = pd.DataFrame([res_lsco])
            df_result.to_csv(results_file, mode='a', header=False, index=False)
            print(".3f")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {dataset_filename}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏁 Benchmark Complete. Processed {processed_count} datasets.")
    print(f"   Results saved to: {results_file}")
    
    # Summary
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n📊 Summary:")
        for method in ['LASSO', 'LSCO']:
            method_results = df_results[df_results['method'] == method]
            if len(method_results) > 0:
                print(f"   {method} NestBoot:")
                print(f"      F1: {method_results['f1'].mean():.3f} ± {method_results['f1'].std():.3f}")
                print(f"      AUROC: {method_results['auroc'].mean():.3f} ± {method_results['auroc'].std():.3f}")
                print(f"      Time: {method_results['time'].mean():.1f} ± {method_results['time'].std():.1f}s")
                print(f"      Density: {method_results['density'].mean():.3f} ± {method_results['density'].std():.3f}")

if __name__ == "__main__":
    main()
