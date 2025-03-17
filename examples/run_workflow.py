#!/usr/bin/env python
"""
Example script demonstrating how to use the NetworkBootstrap workflow.

This script:
1. Creates a workflow directory with the Snakefile and config.yaml
2. Prepares some sample data
3. Runs the Snakemake workflow
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Import workflow functions from the package
try:
    from network_bootstrap import create_workflow_directory, run_workflow
except ImportError:
    print("Make sure to install with workflow extras: pip install -e '.[workflow]'")
    exit(1)

def generate_sample_data(output_dir: Path, sample_name: str, num_genes: int = 50, 
                         num_runs: int = 64) -> tuple:
    """Generate sample network data for demonstration."""
    # Create directories
    data_dir = output_dir / "data" / sample_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate gene names
    genes = [f"gene_{i}" for i in range(num_genes)]
    
    # Generate edges (all possible pairs)
    edges = []
    for i in range(num_genes):
        for j in range(num_genes):
            if i != j:
                edges.append((genes[i], genes[j]))
    
    # Generate normal and shuffled data
    normal_data = []
    shuffled_data = []
    
    for run in range(num_runs):
        for gene_i, gene_j in edges:
            # Normal data: some edges have consistent non-zero values
            if np.random.rand() < 0.2:  # 20% of edges are "real"
                link_value = np.random.normal(0.5, 0.2) 
                normal_data.append({
                    'gene_i': gene_i,
                    'gene_j': gene_j,
                    'run': run,
                    'link_value': link_value
                })
            
            # Shuffled data: random values for all edges
            if np.random.rand() < 0.1:  # 10% of edges in shuffled data
                link_value = np.random.normal(0, 0.3)
                shuffled_data.append({
                    'gene_i': gene_i,
                    'gene_j': gene_j,
                    'run': run,
                    'link_value': link_value
                })
    
    # Convert to DataFrames
    normal_df = pd.DataFrame(normal_data)
    shuffled_df = pd.DataFrame(shuffled_data)
    
    # Save to CSV
    normal_path = data_dir / "normal_data.csv"
    shuffled_path = data_dir / "shuffled_data.csv"
    
    normal_df.to_csv(normal_path, index=False)
    shuffled_df.to_csv(shuffled_path, index=False)
    
    print(f"Generated sample data:")
    print(f"  Normal data: {len(normal_data)} records, saved to {normal_path}")
    print(f"  Shuffled data: {len(shuffled_data)} records, saved to {shuffled_path}")
    
    return normal_path, shuffled_path

def main():
    # Create a workflow directory
    workflow_dir = Path("workflow_example")
    create_workflow_directory(workflow_dir, overwrite=True)
    
    # Create sample data
    samples = ["sampleA", "sampleB"]
    for sample in samples:
        generate_sample_data(workflow_dir, sample, num_genes=20, num_runs=16)
    
    # Update config to use our samples
    config_path = workflow_dir / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    config_content = config_content.replace(
        "samples:\n  - \"sample1\"\n  - \"sample2\"",
        f"samples:\n  - \"{samples[0]}\"\n  - \"{samples[1]}\""
    )
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\nUpdated configuration in {config_path}")
    
    # Run workflow (dry run first)
    print("\nPerforming dry run to check workflow:")
    returncode = run_workflow(workflow_dir, dry_run=True)
    
    if returncode == 0:
        print("\nDry run successful. Running actual workflow:")
        run_workflow(workflow_dir, cores=2)
    else:
        print("\nDry run failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 
