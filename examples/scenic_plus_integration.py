#!/usr/bin/env python
"""
Example script demonstrating how to integrate NetworkBootstrap with SCENIC+.

This script:
1. Loads data from SCENIC+ results
2. Performs bootstrap analysis on SCENIC+ inferred networks
3. Applies NB-FDR to determine significant links
4. Exports and visualizes results
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import scanpy as sc
import mudata as md
import anndata as ad
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("scenic_plus_integration")

try:
    from pyNB import NetworkBootstrap
    import scenicplus as sp
except ImportError:
    logger.error("Make sure to install with workflow extras: pip install -e '.[workflow]'")
    raise

def extract_networks_from_scenicplus(scplus_path: Path, n_runs: int = 64) -> tuple:
    """
    Extract network links from SCENIC+ results.
    
    Args:
        scplus_path: Path to SCENIC+ results
        n_runs: Number of bootstrap runs to simulate
    
    Returns:
        Tuple of (normal_df, shuffled_df) containing network links with bootstrap runs
    """
    logger.info(f"Loading SCENIC+ results from {scplus_path}")
    
    # Load SCENIC+ mudata
    try:
        mdata = md.read(scplus_path)
        logger.info(f"Loaded mudata with modalities: {list(mdata.mod.keys())}")
    except Exception as e:
        logger.error(f"Error loading SCENIC+ data: {e}")
        raise
    
    # Extract network links from eRegulons
    if 'eRegulon_metadata' not in mdata.uns:
        raise ValueError("SCENIC+ data doesn't contain eRegulon metadata")
    
    # Get eRegulon metadata
    try:
        regulons = pd.DataFrame(mdata.uns['eRegulon_metadata'])
        logger.info(f"Found {len(regulons)} eRegulons")
    except Exception as e:
        logger.error(f"Error extracting eRegulons: {e}")
        raise
    
    # Extract TF-gene links
    normal_links = []
    
    # Iterate through eRegulons to extract TF-target links
    for _, regulon in regulons.iterrows():
        tf = regulon['TF']
        target_genes = regulon['extended_target_genes'].split(';')
        weight = regulon['NES']  # Normalized enrichment score
        
        for gene in target_genes:
            if gene and gene != tf:  # Skip self-links
                # Create multiple bootstrap runs with small variations
                for run in range(n_runs):
                    # Add some noise to weights to simulate bootstrap runs
                    run_weight = weight * (1 + np.random.normal(0, 0.1))
                    normal_links.append({
                        'gene_i': tf,
                        'gene_j': gene,
                        'run': run,
                        'link_value': run_weight
                    })
    
    normal_df = pd.DataFrame(normal_links)
    logger.info(f"Extracted {len(set(zip(normal_df['gene_i'], normal_df['gene_j'])))} unique links "
                f"across {n_runs} bootstrap runs")
    
    # Create shuffled version
    shuffled_links = []
    unique_tfs = normal_df['gene_i'].unique()
    unique_targets = normal_df['gene_j'].unique()
    
    # Sample approximately the same number of links but randomized
    n_shuffled = len(set(zip(normal_df['gene_i'], normal_df['gene_j'])))
    
    for run in range(n_runs):
        # Randomly sample TFs and targets
        for _ in range(n_shuffled):
            tf = np.random.choice(unique_tfs)
            gene = np.random.choice(unique_targets)
            if gene != tf:  # Skip self-links
                shuffled_links.append({
                    'gene_i': tf,
                    'gene_j': gene,
                    'run': run,
                    'link_value': np.random.normal(0, 1)
                })
    
    shuffled_df = pd.DataFrame(shuffled_links)
    logger.info(f"Created {len(set(zip(shuffled_df['gene_i'], shuffled_df['gene_j'])))} "
                f"unique shuffled links for comparison")
    
    return normal_df, shuffled_df

def run_nbfdr_on_scenicplus(
    scplus_path: Path,
    output_dir: Path,
    init: int = 64,
    boot: int = 8,
    fdr: float = 0.05
) -> None:
    """
    Run NB-FDR analysis on SCENIC+ results.
    
    Args:
        scplus_path: Path to SCENIC+ results
        output_dir: Directory for output files
        init: Number of bootstrap iterations
        boot: Bootstrap group size
        fdr: False Discovery Rate threshold
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract networks from SCENIC+ results
    normal_df, shuffled_df = extract_networks_from_scenicplus(scplus_path, n_runs=init)
    
    # Save extracted networks for reference
    normal_df.to_csv(output_dir / "normal_links.csv", index=False)
    shuffled_df.to_csv(output_dir / "shuffled_links.csv", index=False)
    
    # Initialize NetworkBootstrap
    nb = NetworkBootstrap()
    
    # Run NB-FDR analysis
    logger.info("Running NB-FDR analysis")
    results = nb.nb_fdr(
        normal_df=normal_df,
        shuffled_df=shuffled_df,
        init=init,
        data_dir=output_dir,
        fdr=fdr,
        boot=boot
    )
    
    # Export results to text file
    nb.export_results(results, output_dir / "results.txt")
    
    # Prepare for plotting
    agg_normal = nb.compute_assign_frac(normal_df, init, boot)
    agg_normal.rename(columns={'Afrac': 'Afrac_norm', 'Asign_frac': 'Asign_frac_norm'}, inplace=True)
    agg_shuffled = nb.compute_assign_frac(shuffled_df, init, boot)
    agg_shuffled.rename(columns={'Afrac': 'Afrac_shuf', 'Asign_frac': 'Asign_frac_shuf'}, inplace=True)
    merged = pd.merge(agg_normal, agg_shuffled, on=['gene_i', 'gene_j'])
    
    # Generate plots
    nb.plot_analysis_results(merged, output_dir / "analysis_plot.png", bins=20)
    
    # Create network file for visualization
    significant_links = merged[merged['Afrac_norm'] >= results.support]
    network_df = pd.DataFrame({
        'source': significant_links['gene_i'],
        'target': significant_links['gene_j'],
        'weight': significant_links['Afrac_norm'],
        'sign': np.sign(significant_links['Asign_frac_norm'])
    })
    network_df.to_csv(output_dir / "filtered_network.csv", index=False)
    
    # Print summary
    print(f"\nNB-FDR Analysis Results:")
    print(f"------------------------")
    print(f"Total links analyzed: {len(merged)}")
    print(f"Significant links at FDR {fdr}: {len(significant_links)}")
    print(f"Support threshold: {results.support:.3f}")
    print(f"False positive rate: {results.fp_rate:.3f}")
    print(f"Results saved to: {output_dir}")
    
    # Return the filtered network and results for further use
    return network_df, results

def main():
    """Main execution function."""
    # Path to SCENIC+ results (change this to your actual path)
    scplus_path = Path("path/to/scenic_plus_results.h5mu")
    
    # Check if the file exists
    if not scplus_path.exists():
        logger.error(f"SCENIC+ results file not found: {scplus_path}")
        logger.info("This is an example script. Please update the file path.")
        
        # Create a dummy output for demonstration
        output_dir = Path("output/scenic_integration_example")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write a note in the output directory
        with open(output_dir / "README.txt", "w") as f:
            f.write("This is a placeholder. To run this example, please update the script with a valid path to SCENIC+ results.")
        
        print(f"Created example output directory at {output_dir}")
        print("This script requires actual SCENIC+ results to run.")
        print("Please update the file path in the script.")
        return
    
    # Set output directory
    output_dir = Path("output/scenic_integration")
    
    # Run NB-FDR analysis on SCENIC+ results
    filtered_network, results = run_nbfdr_on_scenicplus(
        scplus_path=scplus_path,
        output_dir=output_dir,
        init=64,
        boot=8,
        fdr=0.05
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 
