#!/usr/bin/env python
"""
Script to run NB-FDR analysis.

This script is called from the Snakemake workflow.
"""

import pandas as pd
from pathlib import Path
from network_bootstrap.nb_fdr import NetworkBootstrap

# Get inputs, outputs and parameters from Snakemake
input_normal = snakemake.input.normal
input_shuffled = snakemake.input.shuffled
output_results = snakemake.output.results
output_network = snakemake.output.network
fdr = snakemake.params.fdr
init = snakemake.params.init
boot = snakemake.params.boot

# Create output directories if they don't exist
Path(output_results).parent.mkdir(parents=True, exist_ok=True)

# Initialize NetworkBootstrap
nb = NetworkBootstrap()

# Log progress
print(f"Running NB-FDR analysis for {input_normal} and {input_shuffled}")

# Load data
normal_df = pd.read_csv(input_normal)
shuffled_df = pd.read_csv(input_shuffled)

# Rename columns to match expected format for nb_fdr
for df, suffix in [(normal_df, '_norm'), (shuffled_df, '_shuf')]:
    for col in df.columns:
        if col not in ['gene_i', 'gene_j']:
            df.rename(columns={col: f"{col}{suffix}"}, inplace=True)

# Merge datasets
merged = pd.merge(normal_df, shuffled_df, on=['gene_i', 'gene_j'])

# Run NB-FDR analysis
data_dir = Path(output_results).parent
results = nb.nb_fdr(
    normal_df=pd.read_csv(input_normal),
    shuffled_df=pd.read_csv(input_shuffled),
    init=init,
    data_dir=data_dir,
    fdr=fdr,
    boot=boot
)

# Export results to text file
nb.export_results(results, output_results)

# Save network as CSV
network_df = pd.DataFrame({
    'gene_i': [],
    'gene_j': [],
    'weight': [],
    'sign': []
})

# TODO: Populate network_df from results
network_df.to_csv(output_network, index=False)

print(f"NB-FDR analysis completed. Results saved to {output_results}")
