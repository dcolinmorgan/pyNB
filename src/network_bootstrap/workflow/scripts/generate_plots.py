#!/usr/bin/env python
"""
Script to generate analysis plots.

This script is called from the Snakemake workflow.
"""

import pandas as pd
from pathlib import Path
from nb_fdr import NetworkBootstrap

# Get inputs, outputs and parameters from Snakemake
input_normal = snakemake.input.normal
input_shuffled = snakemake.input.shuffled
output_plot = snakemake.output.plot
bins = snakemake.params.bins

# Create output directories if they don't exist
Path(output_plot).parent.mkdir(parents=True, exist_ok=True)

# Initialize NetworkBootstrap
nb = NetworkBootstrap()

# Log progress
print(f"Generating plots for {input_normal} and {input_shuffled}")

# Load data
normal_df = pd.read_csv(input_normal)
shuffled_df = pd.read_csv(input_shuffled)

# Rename columns for merging
normal_df.rename(columns={
    'Afrac': 'Afrac_norm',
    'Asign_frac': 'Asign_frac_norm'
}, inplace=True)
shuffled_df.rename(columns={
    'Afrac': 'Afrac_shuf',
    'Asign_frac': 'Asign_frac_shuf'
}, inplace=True)

# Merge datasets
merged = pd.merge(normal_df, shuffled_df, on=['gene_i', 'gene_j'])

# Generate plot
nb.plot_analysis_results(merged, Path(output_plot), bins=bins)

print(f"Plot generated and saved to {output_plot}")
