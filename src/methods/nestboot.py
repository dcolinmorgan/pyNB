from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def compute_assign_frac(df: pd.DataFrame, total_runs: int = 64, inner_group_size: int = 8) -> pd.DataFrame:
    """
    Compute assignment fractions and signs for network links from bootstrap runs.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw bootstrap records with columns: run, gene_i, gene_j, value
    total_runs : int
        Number of bootstrap runs to consider
    inner_group_size : int
        Size of groups for sign computation
        
    Returns
    -------
    pd.DataFrame
        Aggregated results with columns: gene_i, gene_j, Afrac, Asign_frac
    """
    # Extract run numbers and filter
    df = df[df['run'].str.extract(r'(\d+)').astype(int) < total_runs]
    
    # Group by gene pairs and compute metrics
    grouped = df.groupby(['gene_i', 'gene_j'])
    run_counts = grouped['run'].nunique()
    
    # Initialize results
    results = pd.DataFrame({
        'gene_i': run_counts.index.get_level_values(0),
        'gene_j': run_counts.index.get_level_values(1),
        'Afrac': run_counts / total_runs
    })
    
    # Compute sign fractions for links with full support
    full_support_mask = results['Afrac'] >= 1
    if full_support_mask.any():
        sign_fracs = []
        for (gene_i, gene_j), group in grouped[full_support_mask]:
            # Get one record per run and compute sign fraction
            run_values = (group.drop_duplicates('run')
                         .sort_values('run')['value']
                         .values[:total_runs]
                         .reshape(-1, inner_group_size))
            pos_frac = (run_values > 0).mean(axis=1).mean()
            sign_fracs.append(2 * pos_frac - 1)
            
        results.loc[full_support_mask, 'Asign_frac'] = sign_fracs
    
    # Fill remaining Asign_frac values with 0
    results['Asign_frac'] = results.get('Asign_frac', 0)
    
    return results

def NB_FDR_aggregated(normal_df: pd.DataFrame, 
                      shuffled_df: pd.DataFrame, 
                      support_threshold: float = 0.8, 
                      eps: float = 1e-6) -> pd.DataFrame:
    """
    Compute network comparison metrics between normal and shuffled networks.
    
    Parameters
    ----------
    normal_df : pd.DataFrame
        Normal network data with columns: gene_i, gene_j, Afrac, Asign_frac
    shuffled_df : pd.DataFrame
        Shuffled network data with same columns
    support_threshold : float
        Threshold for binary network
    eps : float
        Small value to prevent division by zero
        
    Returns
    -------
    pd.DataFrame
        Merged results with comparison metrics
    """
    # Merge networks
    merged = pd.merge(
        normal_df, shuffled_df,
        on=["gene_i", "gene_j"],
        suffixes=("_norm", "_shuf")
    )
    
    # Compute all metrics at once
    merged = merged.assign(
        XNETa=(merged["Afrac_norm"] >= support_threshold).astype(float),
        Ssuma=np.sign(merged["Asign_frac_norm"]),
        minAba=merged["Afrac_norm"],
        FF=merged["Afrac_norm"] - merged["Afrac_shuf"],
        FP=merged["Afrac_shuf"] / (merged["Afrac_norm"] + eps),
        supp=merged["Afrac_norm"],
        orig_index=support_threshold
    )
    
    # Compute dependent metrics
    merged["sXNETa"] = merged["XNETa"] * merged["Ssuma"]
    merged["acc"] = merged[["Afrac_norm", "Afrac_shuf"]].values.tolist()
    
    return merged

def plot_network_metrics(merged_df: pd.DataFrame, output_path: str) -> None:
    """Plot network comparison metrics."""
    df_sorted = merged_df.sort_values("Afrac_norm").reset_index(drop=True)
    
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Frequencies
    ax_left.plot(df_sorted["Afrac_norm"], label="Measured", color="dodgerblue")
    ax_left.plot(df_sorted["Afrac_shuf"], label="Null", color="firebrick")
    ax_left.set_title("Link Frequencies")
    ax_left.set_xlabel("Sorted Link Index")
    ax_left.set_ylabel("Frequency")
    ax_left.legend()
    ax_left.set_ylim(0, 1)
    
    # Plot 2: Sign fractions
    ax_right.plot(df_sorted["Afrac_norm"], df_sorted["Asign_frac_norm"], 
                 label="Measured", color="dodgerblue")
    ax_right.plot(df_sorted["Afrac_norm"], df_sorted["Asign_frac_shuf"], 
                 label="Null", color="firebrick")
    ax_right.set_title("Sign Fractions")
    ax_right.set_xlabel("Measured Support")
    ax_right.set_ylabel("Sign Fraction")
    ax_right.legend()
    ax_right.set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_network_data(data_path: str, is_null: bool = False) -> pd.DataFrame:
    """Process raw network data file."""
    df = pd.read_csv(data_path)
    df['runs'] = df.run.str.split('_').str[1].astype(int)
    return df[df['runs'] < 65].sort_values('runs')

def main() -> None:
    """Main execution function."""
    # Process input data
    normal_data = process_network_data('../scenicplus/normal_data.gz')
    null_data = process_network_data('../scenicplus/null_data.gz', is_null=True)
    
    # Compute metrics
    agg_normal = compute_assign_frac(normal_data)
    agg_shuffled = compute_assign_frac(null_data)
    
    # Rename columns
    suffix_map = {
        'normal': ('_norm', agg_normal),
        'shuffled': ('_shuf', agg_shuffled)
    }
    for name, (suffix, df) in suffix_map.items():
        df.rename(columns={
            'Afrac': f'Afrac{suffix}',
            'Asign_frac': f'Asign_frac{suffix}'
        }, inplace=True)
    
    # Compute and save results
    results = NB_FDR_aggregated(agg_normal, agg_shuffled)
    results.to_csv("NB_FDR_results.csv", index=False)
    
    # Generate plots
    plot_network_metrics(results, "network_metrics.png")

if __name__ == '__main__':
    main()
    
