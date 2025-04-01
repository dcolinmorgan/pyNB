import pandas as pd
from pathlib import Path
from src.nb_fdr import NetworkBootstrap

def process_network_data(data_path: str, is_null: bool = False) -> pd.DataFrame:
    """Process raw network data file.
    
    Args:
        data_path: Path to data file
        is_null: Whether this is null/shuffled data
        
    Returns:
        Processed DataFrame
    """
    df = pd.read_csv(data_path)
    df['run'] = df.run.str.extract(r'(\d+)').astype(int)
    return df[df['run'] < 65].sort_values('run')

def main() -> None:
    """Main execution function."""
    # Process input data
    normal_data = process_network_data('../scenicplus/normal_data.gz')
    null_data = process_network_data('../scenicplus/null_data.gz', is_null=True)
    
    # Initialize analyzer
    nb = NetworkBootstrap()
    
    # Run analysis
    results = nb.nb_fdr(
        normal_df=normal_data,
        shuffled_df=null_data,
        method="example",
        init=64,
        data_dir=Path("output"),
        fdr=0.05,
        boot=8
    )
    
    # Print results
    print(f"Network sparsity: {(results.xnet != 0).mean():.3f}")
    print(f"Node count: {results.xnet.shape[0]:.3f}")
    print(f"Edge count: {results.xnet.sum():.3f}")
    print(f"False positive rate: {results.fp_rate:.3f}")
    print(f"Support threshold: {results.support:.3f}")

    # Export results to a text file
    nb.export_results(results, Path("output/results.txt"))
    
    # To generate the analysis plot, re-create the merged DataFrame (as done in nb_fdr)
    agg_normal = nb.compute_assign_frac(normal_data, 64, 8)
    agg_normal.rename(columns={'Afrac': 'Afrac_norm', 'Asign_frac': 'Asign_frac_norm'}, inplace=True)
    agg_shuffled = nb.compute_assign_frac(null_data, 64, 8)
    agg_shuffled.rename(columns={'Afrac': 'Afrac_shuf', 'Asign_frac': 'Asign_frac_shuf'}, inplace=True)
    merged = pd.merge(agg_normal, agg_shuffled, on=['gene_i', 'gene_j'])
    
    nb.plot_analysis_results(merged, Path("output/analysis_plot.png"), bins=32)

if __name__ == '__main__':
    main() 
