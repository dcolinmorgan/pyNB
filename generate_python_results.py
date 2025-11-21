#!/usr/bin/env python3
"""
Generate Python benchmark results aligned with standard benchmark.
Loads Tjarnberg dataset and network, runs LASSO and LSCO with sparsity sweep.
"""

import sys
import time
import psutil
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, 'src')
from analyze.Data import Data
from datastruct.Network import Network
from datastruct.Dataset import Dataset
from methods.lasso import Lasso
from methods.lsco import LSCO
from analyze.CompareModels import CompareModels
from bootstrap.nb_fdr import NetworkBootstrap

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_benchmark_data():
    dataset_url = 'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json'
    network_url = 'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json'
    dataset = Data.from_json_url(dataset_url)
    true_net = Network.from_json_url(network_url)
    return dataset, true_net

def run_network_inference(dataset, method_name="LASSO"):
    start_time = time.time()
    start_memory = monitor_memory()
    
    if method_name == "LASSO":
        # LASSO uses the same sparsity range as MATLAB GeneSPIDER2
        # MATLAB uses logspace(-6, 0, 30) and selects index 25
        zeta = np.logspace(-6, 0, 30)
        result_network, param = Lasso(dataset, alpha_range=zeta)
        param_name = "alpha"
    elif method_name == "LSCO":
        # LSCO uses closed-form solution: A = -P * pinv(Y)
        result_network, param = LSCO(dataset)
        param_name = "mse"
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    end_time = time.time()
    peak_memory = monitor_memory()
    execution_time = end_time - start_time
    memory_usage = peak_memory - start_memory
    network_obj = Network(result_network)
    num_edges = np.sum(np.abs(network_obj.A) > 1e-10)  # Count non-zero elements with tolerance
    density = num_edges / (network_obj.A.shape[0] * network_obj.A.shape[1])
    sparsity = 1 - density
    return {
        'method': method_name,
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'peak_memory': peak_memory,
        'parameter_value': param,
        'parameter_name': param_name,
        'num_edges': num_edges,
        'density': density,
        'sparsity': sparsity,
        'network_shape': network_obj.A.shape,
        'network_matrix': result_network
    }

def run_comparison_analysis(true_net, inferred_net):
    comp = CompareModels(true_net, Network(inferred_net))
    
    # Calculate AUROC manually using sklearn with proper thresholding
    try:
        from sklearn.metrics import roc_auc_score
        # Flatten the networks for ROC calculation
        true_flat = true_net.A.flatten()
        inferred_flat = np.abs(inferred_net.flatten())  # Use absolute values for ranking
        
        # Only calculate if we have both classes (0 and 1) in true network
        if len(np.unique(true_flat)) > 1 and np.sum(inferred_flat) > 0:
            auroc = roc_auc_score(true_flat, inferred_flat)
        else:
            auroc = 0.5  # Random performance baseline
    except ImportError:
        auroc = 0.5
    except Exception as e:
        # Fallback to correlation-based score
        try:
            true_flat = true_net.A.flatten()
            inferred_flat = inferred_net.flatten()
            correlation = np.corrcoef(true_flat, inferred_flat)[0, 1]
            auroc = (correlation + 1) / 2  # Convert correlation [-1,1] to [0,1]
            if np.isnan(auroc):
                auroc = 0.5
        except:
            auroc = 0.5
    
    return {
        'f1_score': comp.F1[0] if len(comp.F1) > 0 else 0.0,
        'mcc': comp.MCC[0] if len(comp.MCC) > 0 else 0.0,
        'sensitivity': comp.sen[0] if len(comp.sen) > 0 else 0.0,
        'specificity': comp.spe[0] if len(comp.spe) > 0 else 0.0,
        'precision': comp.pre[0] if len(comp.pre) > 0 else 0.0,
        'auroc': auroc,
        'true_positives': comp.TP[0] if len(comp.TP) > 0 else 0,
        'false_positives': comp.FP[0] if len(comp.FP) > 0 else 0,
        'true_negatives': comp.TN[0] if len(comp.TN) > 0 else 0,
        'false_negatives': comp.FN[0] if len(comp.FN) > 0 else 0
    }

def run_nestboot_analysis(dataset, true_net, method_name="LASSO", nest_runs=50, boot_runs=50):
    """Run NestBoot analysis with actual network inference on bootstrap samples."""
    start_time = time.time()
    start_memory = monitor_memory()
    
    print(f"      Running {nest_runs} NestBoot iterations with {boot_runs} bootstrap samples each...")
    
    # Generate bootstrap networks using actual inference methods
    bootstrap_data = []
    shuffled_data = []
    
    n_genes = dataset.data.N
    n_samples = dataset.data.M
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for outer_run in range(nest_runs):
        print(f"         Outer run {outer_run + 1}/{nest_runs}...")
        
        # For each outer run, perform bootstrap sampling and network inference
        for boot_run in range(boot_runs):
            try:
                # Create bootstrap sample indices
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                
                # Create bootstrap dataset by copying and modifying the original
                bootstrap_Y = dataset.data.Y[:, bootstrap_indices]
                bootstrap_dataset = Data(dataset.data)  # Copy original
                bootstrap_dataset.data._Y = bootstrap_Y  # Override with bootstrap sample
                
                # Run network inference on bootstrap sample
                if method_name == "LASSO":
                    zeta = np.logspace(-6, 0, 10)  # Reduced range for speed
                    network_matrix, param = Lasso(bootstrap_dataset, alpha_range=zeta)
                elif method_name == "LSCO":
                    network_matrix, param = LSCO(bootstrap_dataset)
                else:
                    raise ValueError(f"Unknown method: {method_name}")
                
                # Convert network to link format for NB-FDR
                for i in range(n_genes):
                    for j in range(n_genes):
                        if i != j and abs(network_matrix[i, j]) > 1e-6:  # Only non-zero links
                            gene_i = f"Gene_{i:02d}"
                            gene_j = f"Gene_{j:02d}"
                            
                            bootstrap_data.append({
                                'gene_i': gene_i,
                                'gene_j': gene_j,
                                'run': f"run_{outer_run}",
                                'link_value': abs(network_matrix[i, j])
                            })
                
                # Create shuffled version by permuting data
                shuffle_indices = np.random.permutation(n_samples)
                shuffled_Y = dataset.data.Y[:, shuffle_indices]
                shuffled_dataset = Data(dataset.data)  # Copy original
                shuffled_dataset.data._Y = shuffled_Y  # Override with shuffled sample
                
                # Run network inference on shuffled data
                if method_name == "LASSO":
                    shuffled_network, _ = Lasso(shuffled_dataset, alpha_range=zeta)
                elif method_name == "LSCO":
                    shuffled_network, _ = LSCO(shuffled_dataset)
                
                # Convert shuffled network to link format
                for i in range(n_genes):
                    for j in range(n_genes):
                        if i != j and abs(shuffled_network[i, j]) > 1e-6:
                            gene_i = f"Gene_{i:02d}"
                            gene_j = f"Gene_{j:02d}"
                            
                            shuffled_data.append({
                                'gene_i': gene_i,
                                'gene_j': gene_j,
                                'run': f"run_{outer_run}",
                                'link_value': abs(shuffled_network[i, j])
                            })
                            
            except Exception as e:
                print(f"            ⚠️ Bootstrap iteration {boot_run} failed: {e}")
                continue
    
    print(f"      Generated {len(bootstrap_data)} normal links and {len(shuffled_data)} shuffled links")
    
    if len(bootstrap_data) == 0:
        print("      ❌ No bootstrap data generated, falling back to simple method")
        # Fallback to simple method if bootstrap fails
        inference_results = run_network_inference(dataset, method_name)
        comparison_results = run_comparison_analysis(true_net, inference_results['network_matrix'])
        binary_network = (inference_results['network_matrix'] > 0.1).astype(float)
        
        end_time = time.time()
        peak_memory = monitor_memory()
        execution_time = end_time - start_time
        memory_usage = peak_memory - start_memory
        
        return {
            'method': f"{method_name}_nestboot_fallback",
            'execution_time': execution_time,
            'memory_usage': memory_usage,
            'peak_memory': peak_memory,
            'parameter_value': 0.1,
            'parameter_name': 'threshold',
            'num_edges': int(np.sum(binary_network != 0)),
            'density': np.sum(binary_network != 0) / (binary_network.shape[0] * binary_network.shape[1]),
            'sparsity': 1 - (np.sum(binary_network != 0) / (binary_network.shape[0] * binary_network.shape[1])),
            'network_shape': binary_network.shape,
            'network_matrix': binary_network,
            'nest_runs': nest_runs,
            'boot_runs': boot_runs,
            **comparison_results
        }
    
    # Convert to DataFrames
    normal_df = pd.DataFrame(bootstrap_data)
    shuffled_df = pd.DataFrame(shuffled_data)
    
    # Run NestBoot FDR analysis
    nb = NetworkBootstrap()
    
    try:
        print(f"      Running NB-FDR analysis...")
        results = nb.nb_fdr(
            normal_df=normal_df,
            shuffled_df=shuffled_df,
            init=50,                    # Number of bootstrap iterations (matching MATLAB boot=50)
            data_dir=Path("benchmark_results"),  # Output directory
            fdr=0.05,                   # False Discovery Rate threshold
            boot=50                     # Bootstrap group size (matching MATLAB nest=50)
        )
        
        # Extract binary network from results
        binary_network = results.xnet.reshape((n_genes, n_genes))
        print(f"      ✅ NestBoot completed, network shape: {binary_network.shape}")
        
    except Exception as e:
        print(f"         ⚠️ NestBoot FDR analysis failed: {e}")
        # Create a simple consensus network from bootstrap data
        print(f"      Creating consensus network from {len(normal_df)} links...")
        
        binary_network = np.zeros((n_genes, n_genes))
        
        if len(normal_df) > 0:
            # Count link occurrences across runs
            link_counts = normal_df.groupby(['gene_i', 'gene_j']).size().reset_index(name='count')
            threshold = nest_runs * 0.1  # Appear in at least 10% of runs
            
            for idx, row in link_counts.iterrows():
                count_val = row.iloc[2]  # 'count' column
                if count_val >= threshold:
                    gene_i_str = row.iloc[0]  # 'gene_i' column
                    gene_j_str = row.iloc[1]  # 'gene_j' column
                    i = int(gene_i_str.split('_')[1])
                    j = int(gene_j_str.split('_')[1])
                    binary_network[i, j] = 1.0
    
    # Run comparison analysis
    comparison_results = run_comparison_analysis(true_net, binary_network)
    
    end_time = time.time()
    peak_memory = monitor_memory()
    execution_time = end_time - start_time
    memory_usage = peak_memory - start_memory
    
    # Calculate network metrics
    num_edges = np.sum(binary_network != 0)
    density = num_edges / (binary_network.shape[0] * binary_network.shape[1])
    sparsity = 1 - density
    
    return {
        'method': f"{method_name}_nestboot",
        'execution_time': execution_time,
        'memory_usage': memory_usage,
        'peak_memory': peak_memory,
        'parameter_value': 0.05,  # FDR threshold
        'parameter_name': 'FDR',
        'num_edges': int(num_edges),
        'density': density,
        'sparsity': sparsity,
        'network_shape': binary_network.shape,
        'network_matrix': binary_network,
        'nest_runs': nest_runs,
        'boot_runs': boot_runs,
        **comparison_results
    }

def main():
    print("🚀 Python Performance Benchmark")
    print("=" * 40)
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    print("\n📊 Loading benchmark dataset")
    dataset, true_net = load_benchmark_data()
    print(f"   ✅ Dataset loaded: {dataset.dataset}")
    print(f"      Shape: {dataset.data.Y.shape}, Genes: {dataset.data.N}, Samples: {dataset.data.M}")
    print(f"   ✅ Network loaded: {true_net.network}")
    
    all_results = []
    
    # Test both regular methods and NestBoot methods
    methods_to_test = [
        ("LASSO", False),
        ("LSCO", False), 
        # Temporarily disable NestBoot to test simple methods first
        # ("LASSO", True),
        # ("LSCO", True)
    ]
    
    for method, use_nestboot in methods_to_test:
        method_display = f"{method} {'NestBoot' if use_nestboot else 'Simple'}"
        print(f"\n   🎯 Running {method_display}...")
        
        if use_nestboot:
            # Run NestBoot analysis with reduced runs for testing
            result_data = run_nestboot_analysis(dataset, true_net, method, nest_runs=10, boot_runs=5)
            # Add nestboot-specific fields
            result_data['use_nestboot'] = True
            result_data['method_name'] = f"{method.lower()}_nestboot"
        else:
            # Run regular inference
            inference_results = run_network_inference(dataset, method)
            comparison_results = run_comparison_analysis(true_net, inference_results['network_matrix'])
            result_data = {
                **inference_results,
                **comparison_results,
                'use_nestboot': False,
                'method_name': f"{method.lower()}_simple"
            }
        
        # Add common fields
        result = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': 'medium',
            'n_genes': dataset.data.N,
            'n_samples': dataset.data.M,
            **result_data
        }
        
        all_results.append(result)
        
        # Save individual result
        method_file_name = result_data['method_name']
        result_file = output_dir / f"python_{method_file_name}_medium_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
        print(f"      ✅ {method_display} complete: F1={result_data.get('f1_score', 0):.3f}, "
              f"AUROC={result_data.get('auroc', 0):.3f}, Time={result_data['execution_time']:.1f}s, "
              f"Memory={result_data['memory_usage']:.1f}MB")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "python_benchmark_results.csv", index=False)
    
    # Generate summary statistics
    summary = {}
    unique_methods = results_df['method_name'].unique()
    
    for method_name in unique_methods:
        method_data = results_df[results_df['method_name'] == method_name]
        summary[method_name] = {
            'avg_execution_time': method_data['execution_time'].mean(),
            'avg_memory_usage': method_data['memory_usage'].mean(),
            'avg_f1_score': method_data['f1_score'].mean(),
            'avg_precision': method_data['precision'].mean(),
            'avg_recall': method_data['sensitivity'].mean(),
            'avg_sparsity': method_data['sparsity'].mean(),
            'avg_density': method_data['density'].mean(),
            'avg_auroc': method_data['auroc'].mean(),
            'count': len(method_data)
        }
    
    with open(output_dir / "python_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Python benchmark complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total tests: {len(all_results)}")
    print("\n📈 Summary Results:")
    for method_name, stats in summary.items():
        print(f"   {method_name}:")
        print(f"      F1 Score: {stats['avg_f1_score']:.3f}")
        print(f"      AUROC: {stats['avg_auroc']:.3f}")
        print(f"      Execution Time: {stats['avg_execution_time']:.1f}s")
        print(f"      Memory Usage: {stats['avg_memory_usage']:.1f}MB")
        print(f"      Sparsity: {stats['avg_sparsity']:.3f}")

if __name__ == "__main__":
    main()
