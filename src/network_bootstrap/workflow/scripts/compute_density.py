#!/usr/bin/env python
"""
Script to compute network density.

This script is called from the Snakemake workflow.
"""

import pandas as pd
from pathlib import Path
from network_bootstrap.nb_fdr import NetworkBootstrap

# Get inputs, outputs and parameters from Snakemake
input_normal = snakemake.input.normal
output_density = snakemake.output.density
threshold = snakemake.params.threshold

# Create output directories if they don't exist
Path(output_density).parent.mkdir(parents=True, exist_ok=True)

# Initialize NetworkBootstrap
nb = NetworkBootstrap()

# Log progress
print(f"Computing network density for {input_normal}")

# Load data
normal_df = pd.read_csv(input_normal)

# Ensure run column is properly formatted
if 'run' not in normal_df.columns and 'runs' in normal_df.columns:
    normal_df.rename(columns={'runs': 'run'}, inplace=True)

# If run is a string with format like 'run_5', extract the number
if normal_df['run'].dtype == 'object':
    normal_df['run'] = normal_df.run.str.extract(r'(\d+)').astype(int)

# If link_value column has different name, rename it
if 'link_value' not in normal_df.columns and 'value' in normal_df.columns:
    normal_df.rename(columns={'value': 'link_value'}, inplace=True)

# Compute network density
density_df = nb.compute_network_density(normal_df, threshold=threshold)

# Save to CSV
density_df.to_csv(output_density, index=False)

print(f"Network density data saved to {output_density}")
