from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, TypeVar, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from pathlib import Path
from .utils import NetworkUtils

# Import configuration
try:
    from ..config import AnalysisConfig
except ImportError:
    # Fallback if config module not available
    @dataclass
    class AnalysisConfig:
        total_runs: int = 64
        inner_group_size: int = 8
        support_threshold: float = 0.8
        fdr_threshold: float = 0.05
        epsilon: float = 1e-10

NDArrayFloat = npt.NDArray[np.float64]
NDArrayBool = npt.NDArray[np.bool_]

@dataclass
class NetworkData:
    """Data class to hold network analysis data.

    Attributes:
        Y: Input data matrix
        names: Node names
        N: Number of nodes
        M: Number of measurements/samples
    """
    Y: NDArrayFloat
    names: List[str]
    N: int
    M: int

@dataclass
class NetworkResults:
    """Results from network bootstrap analysis.

    Attributes:
        xnet: Final network adjacency matrix
        ssum: Sum of sign support
        min_ab: Minimum absolute values
        sxnet: Sign-specific network
        orig_index: Original index
        accumulated: Accumulated statistics
        binned_freq: Binned frequencies
        fp_rate: False positive rate at crossing
        support: Support at crossing
    """
    xnet: NDArrayFloat
    ssum: NDArrayFloat
    min_ab: NDArrayFloat
    sxnet: NDArrayFloat
    orig_index: int
    accumulated: NDArrayFloat
    binned_freq: NDArrayFloat
    fp_rate: float
    support: float

class NetworkBootstrap:
    """Class for performing Network Bootstrap False Discovery Rate analysis.
    
    This class implements the NB-FDR algorithm for network inference with 
    bootstrap-based confidence estimation.
    """

    def __init__(self, param: Optional[Union[logging.Logger, NetworkData, AnalysisConfig, dict]] = None) -> None:
        """Initialize NetworkBootstrap analyzer.

        Args:
            param: Optional parameter which can be:
                   - logger instance
                   - NetworkData object  
                   - AnalysisConfig object
                   - dict with configuration parameters
                   - None (uses defaults)
        """
        # Initialize configuration
        if isinstance(param, AnalysisConfig):
            self.config = param
            self.logger = logging.getLogger(__name__)
            self.data = None
        elif isinstance(param, dict):
            self.config = AnalysisConfig(**param)
            self.logger = logging.getLogger(__name__)
            self.data = None
        elif isinstance(param, logging.Logger):
            self.config = AnalysisConfig()
            self.logger = param
            self.data = None
        elif isinstance(param, NetworkData):
            self.config = AnalysisConfig()
            self.data = param
            self.logger = logging.getLogger(__name__)
        elif param is None:
            self.config = AnalysisConfig()
            self.logger = logging.getLogger(__name__)
            self.data = None
        else:
            raise TypeError("Invalid type for parameter. Expected AnalysisConfig, dict, logging.Logger, or NetworkData.")
        
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging if no logger was provided."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def compute_assign_frac(
        self, 
        df: pd.DataFrame, 
        total_runs: int = 64, 
        inner_group_size: int = 8
    ) -> pd.DataFrame:
        """Compute assignment fractions and signs for network links.

        Args:
            df: DataFrame with columns: gene_i, gene_j, run, link_value
            total_runs: Number of bootstrap runs
            inner_group_size: Size of groups for sign computation

        Returns:
            DataFrame with computed metrics
        """
        self.logger.debug("Computing assignment fractions")
        
        # Extract run numbers and filter out runs greater than or equal to total_runs
        df = df[df['run'].astype(int) < total_runs]
        
        # Group by gene pairs ensuring columns remain in the DataFrame
        grouped = df.groupby(['gene_i', 'gene_j'], as_index=False)
        run_counts = grouped['run'].nunique()
        run_counts['Afrac'] = run_counts['run'] / total_runs
        # Use drop to remove the temporary 'run' count column
        results = run_counts.drop(columns=['run'])
        
        # Compute sign fractions for links with full support (Afrac >= 1)
        full_support = results[results['Afrac'] >= 1]
        sign_fracs = {}
        # Iterate over full-support rows; for each row, get the corresponding group
        for _, row in full_support.iterrows():
            gene_i = row['gene_i']
            gene_j = row['gene_j']
            # Get the group via the tuple key
            group = df[(df['gene_i'] == gene_i) & (df['gene_j'] == gene_j)]
            # Ensure unique run values and sort by run number
            run_values = group.drop_duplicates('run').sort_values('run')['link_value'].values
            # Limit to the first total_runs elements
            r = run_values[:total_runs]
            # Determine the number of complete inner groups 
            num_groups = len(r) // inner_group_size
            if num_groups > 0:
                # Trim the array so it divides evenly
                r = r[: num_groups * inner_group_size]
                r = r.reshape(num_groups, inner_group_size)
                pos_frac = (r > 0).mean(axis=1).mean()
                sign = 2 * pos_frac - 1
            else:
                sign = 0
            sign_fracs[(gene_i, gene_j)] = sign
        
        # Assign the sign fraction column using the calculated dictionary;
        # default to zero if not present
        results['Asign_frac'] = results.apply(
            lambda row: sign_fracs.get((row['gene_i'], row['gene_j']), 0), axis=1
        )
        return results

    def nb_fdr(
        self,
        normal_df: pd.DataFrame,
        shuffled_df: pd.DataFrame,
        init: int,
        data_dir: Path,
        fdr: float,
        boot: int
    ) -> NetworkResults:
        """Perform Network Bootstrap FDR analysis.

        Args:
            normal_df: Normal network data with gene_i, gene_j, run, link_value columns
            shuffled_df: Shuffled network data with same columns
            init: Number of initialization iterations
            data_dir: Directory for output files
            fdr: False Discovery Rate threshold
            boot: Number of bootstrap iterations

        Returns:
            NetworkResults object containing analysis results
        """
        self.logger.info("Starting NB-FDR analysis")
        
        # Compute assignment fractions 
        agg_normal = self.compute_assign_frac(normal_df, init, boot)
        agg_shuffled = self.compute_assign_frac(shuffled_df, init, boot)
        
        # Rename columns for merging
        for df, suffix in [(agg_normal, '_norm'), (agg_shuffled, '_shuf')]:
            df.rename(columns={
                'Afrac': f'Afrac{suffix}',
                'Asign_frac': f'Asign_frac{suffix}'
            }, inplace=True)
        
        # Merge and compute metrics
        merged = pd.merge(
            agg_normal, agg_shuffled,
            on=['gene_i', 'gene_j']
        )
        
        support_threshold = 0.8  # Can be made parameter if needed
        results = self._compute_network_metrics(merged, support_threshold)
        
        self.logger.info("NB-FDR analysis completed successfully")
        return results

    def _compute_network_metrics(
        self, 
        merged: pd.DataFrame,
        support_threshold: float
    ) -> NetworkResults:
        """Compute network comparison metrics.

        Args:
            merged: Merged normal and shuffled network data
            support_threshold: Threshold for binary network

        Returns:
            NetworkResults object
        """
        eps = 1e-6  # Small value to prevent division by zero
        
        # Compute metrics
        xnet = (merged['Afrac_norm'] >= support_threshold).astype(float)
        ssum = np.sign(merged['Asign_frac_norm'])
        min_ab = merged['Afrac_norm']
        sxnet = xnet * ssum
        
        # Compute additional metrics
        ff = merged['Afrac_norm'] - merged['Afrac_shuf']
        fp = merged['Afrac_shuf'] / (merged['Afrac_norm'] + eps)
        
        # Compute accumulated statistics and frequencies
        accumulated = self._compute_accumulated_stats(merged)
        binned_freq = self._compute_binned_frequencies(merged)
        
        return NetworkResults(
            xnet=xnet.values,
            ssum=ssum.values,
            min_ab=min_ab.values,
            sxnet=sxnet.values,
            orig_index=int(support_threshold * 100),
            accumulated=accumulated,
            binned_freq=binned_freq,
            fp_rate=fp.mean(),
            support=support_threshold
        )

    def _accumulate(
        self,
        boo_alink: List[NDArrayFloat],
        boo_shuffle_alink: List[NDArrayFloat],
        init: int
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, 
               NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Accumulate network statistics from bootstrap samples.

        Args:
            boo_alink: Bootstrap network samples
            boo_shuffle_alink: Shuffled bootstrap samples
            init: Number of iterations

        Returns:
            Tuple containing:
            - accumulated: Accumulated statistics
            - sup_over: Support overlap
            - shu_over: Shuffle overlap
            - overlaps_support: Support overlaps
            - overlaps_shuffle: Shuffle overlaps
            - freq: Frequency statistics
        """
        self.logger.debug("Accumulating network statistics")
        
        estimated_support_net: List[NDArrayFloat] = []
        estimated_shuffle_net: List[NDArrayFloat] = []
        overlaps_support: List[NDArrayFloat] = []
        overlaps_shuffle: List[NDArrayFloat] = []

        for i in range(len(boo_alink)):
            est_net, ovr_sup = self._structure_boot(
                boo_alink, i, estimated_support_net, init
            )
            est_shuf, ovr_shuf = self._structure_boot(
                boo_shuffle_alink, i, estimated_shuffle_net, init
            )
            
            estimated_support_net.extend(est_net)
            overlaps_support.extend(ovr_sup)
            estimated_shuffle_net.extend(est_shuf)
            overlaps_shuffle.extend(ovr_shuf)

        freq = np.concatenate([estimated_support_net, estimated_shuffle_net])

        sup_over = np.zeros(init + 1)
        shu_over = np.zeros(init + 1)

        for k in range(init):
            sup_over[k] = self._structure_support(overlaps_support, k, init)
            shu_over[k] = self._structure_support(overlaps_shuffle, k, init)

        # Replace NaN values with 0
        sup_over = np.nan_to_num(sup_over)
        shu_over = np.nan_to_num(shu_over)
        
        accumulated = np.column_stack([sup_over, shu_over])
        
        return accumulated, sup_over, shu_over, np.array(overlaps_support), \
               np.array(overlaps_shuffle), freq

    def _structure_boot(
        self,
        boot_links: List[NDArrayFloat],
        idx: int,
        estimated_net: List[NDArrayFloat],
        init: int
    ) -> Tuple[List[NDArrayFloat], List[NDArrayFloat]]:
        """Process bootstrap samples for network structure.

        Args:
            boot_links: List of bootstrap network samples
            idx: Current index
            estimated_net: List to store estimated network values
            init: Number of iterations

        Returns:
            Tuple containing estimated network values and overlaps
        """
        tmp = boot_links[idx]
        estimated_net.append(tmp.flatten())
        overlaps = [tmp.flatten()]
        return estimated_net, overlaps

    def _structure_support(
        self,
        overlaps: List[NDArrayFloat],
        k: int,
        init: int
    ) -> float:
        """Calculate structure support statistics.

        Args:
            overlaps: List of overlap matrices
            k: Current iteration index
            init: Number of iterations

        Returns:
            Support statistic value
        """
        threshold = k / init
        overlaps_array = np.array(overlaps)
        intersect = np.sum(NetworkUtils.matrix_and(overlaps_array >= threshold))
        union = np.sum(NetworkUtils.matrix_or(overlaps_array >= threshold))
        return intersect / union if union != 0 else 0.0

    def _get_plottable_results(
        self,
        freq: NDArrayFloat,
        init: int,
        accumulated: NDArrayFloat,
        overlaps_support: NDArrayFloat,
        fdr: float
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat,
               float, float, int]:
        """Generate plottable results from analysis.

        Args:
            freq: Frequency statistics
            init: Number of iterations
            accumulated: Accumulated statistics
            overlaps_support: Support overlaps
            fdr: False Discovery Rate threshold

        Returns:
            Tuple containing plot-ready results
        """
        binned_freq = NetworkUtils.calc_bin_freq(freq, init)[0]
        y_range = np.array([0, 1])  # Placeholder for actual y-range calculation

        orig_index = self._find_fdr_cutoff(binned_freq, accumulated, fdr)
        
        support_threshold = (orig_index - 1) / init
        final_net = NetworkUtils.matrix_and(overlaps_support >= support_threshold)
        
        overlap_100 = accumulated[-1, :]
        overlap_cross = accumulated[orig_index, :]
        support_cross = support_threshold
        
        tmp_sum = np.sum(binned_freq[orig_index:], axis=0)
        fp_rate_cross = tmp_sum[1] / tmp_sum[0] if tmp_sum[0] != 0 else 0.0

        return (y_range, overlap_100, final_net, overlap_cross,
                support_cross, fp_rate_cross, orig_index)

    def _compute_accumulated_stats(self, merged: pd.DataFrame) -> NDArrayFloat:
        """Compute accumulated statistics from merged results.
        
        This implementation computes cumulative sums of 'Afrac_norm' and 'Afrac_shuf'
        in the merged DataFrame after sorting by 'Afrac_norm'.
        
        Args:
            merged: Merged DataFrame containing 'Afrac_norm' and 'Afrac_shuf' columns.
        
        Returns:
            A 2D numpy array containing the accumulated stats.
        """
        sorted_df = merged.sort_values('Afrac_norm')
        cum_sum_norm = sorted_df['Afrac_norm'].cumsum().to_numpy()
        cum_sum_shuf = sorted_df['Afrac_shuf'].cumsum().to_numpy()
        return np.column_stack((cum_sum_norm, cum_sum_shuf))
    
    def _compute_binned_frequencies(self, merged: pd.DataFrame, bins: int = 10) -> NDArrayFloat:
        """Compute binned frequencies for the 'Afrac_norm' values.
        
        Args:
            merged: Merged DataFrame from which to compute the histogram.
            bins: Number of bins to use.
        
        Returns:
            A normalized frequency histogram as a numpy array.
        """
        hist, _ = np.histogram(merged['Afrac_norm'], bins=bins, range=(0, 1))
        if hist.sum() > 0:
            return hist.astype(float) / hist.sum()
        return hist.astype(float)

    def export_results(self, results: NetworkResults, txt_file: Path) -> None:
        """Export analysis results to a text file.

        Args:
            results: NetworkResults object
            txt_file: Path to the text file to be written
        """
        with open(txt_file, 'w') as f:
            f.write("Network Bootstrap FDR Analysis Results\n")
            f.write("="*40 + "\n")
            f.write(f"Orig Index: {results.orig_index}\n")
            f.write(f"FP Rate: {results.fp_rate:.3f}\n")
            f.write(f"Support Threshold: {results.support:.3f}\n")
            f.write(f"xnet shape: {results.xnet.shape}\n")
            f.write(f"ssum shape: {results.ssum.shape}\n")
            f.write(f"min_ab shape: {results.min_ab.shape}\n")
            f.write(f"sxnet shape: {results.sxnet.shape}\n")
            f.write("Accumulated (first 5 rows):\n")
            np.savetxt(f, results.accumulated[:5], fmt='%.4f')
            f.write("\nBinned frequencies:\n")
            np.savetxt(f, results.binned_freq[np.newaxis, :], fmt='%.4f')
    
    def plot_analysis_results(self, merged: pd.DataFrame, plot_file: Path, bins: int = 10) -> None:
        """Plot analysis results with link frequencies for normal and shuffled data.

        Args:
            merged: Merged DataFrame with 'Afrac_norm' and 'Afrac_shuf' columns.
            plot_file: Path to save the plot image.
            bins: Number of bins for support.
        """
        # Bin data
        support_bins = np.linspace(0, 1, bins + 1)
        bin_centers = (support_bins[:-1] + support_bins[1:]) / 2
        counts_norm, _ = np.histogram(merged['Afrac_norm'], bins=support_bins)
        counts_shuf, _ = np.histogram(merged['Afrac_shuf'], bins=support_bins)
        freq_norm = counts_norm.astype(float) / counts_norm.sum() if counts_norm.sum() > 0 else counts_norm.astype(float)
        freq_shuf = counts_shuf.astype(float) / counts_shuf.sum() if counts_shuf.sum() > 0 else counts_shuf.astype(float)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))

        # Line plots with distinct styles
        ax.plot(bin_centers, freq_norm, color='#1b9e77', marker='o', linestyle='-', linewidth=2, label='Normal Data')
        ax.plot(bin_centers, freq_shuf, color='#d95f02', marker='^', linestyle='--', linewidth=2, label='Shuffled Data')

        # Aesthetics and readability
        ax.set_xlabel('Support', fontsize=12)
        ax.set_ylabel('Link Frequency', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)  # Light grid

        # Highlight max difference
        diff = freq_norm - freq_shuf
        max_diff_idx = np.argmax(np.abs(diff))
        ax.annotate(
            f'Max Diff: {diff[max_diff_idx]:.2f}',
            xy=(bin_centers[max_diff_idx], max(freq_norm[max_diff_idx], freq_shuf[max_diff_idx])),
            xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray')
        )

        # Legend with title
        ax.legend(title='Data Type', loc='upper right', fontsize=10, title_fontsize=12)

        # Optional: Add support threshold (e.g., 0.8 from your code)
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.8)')
        # if 'Threshold (0.8)' not in [l.get_label() for l in ax.get_legend_handlers_labels()[1]]:
            # ax.legend(title='Data Type', loc='upper right', fontsize=10, title_fontsize=12)

        fig.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')  # Ensure annotations fit
        plt.close()

    def compute_network_density(self, df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """Compute network density per run.

        Network density is calculated as the number of links (edges) between gene_i and gene_j
        divided by the total number of possible edges among unique genes, per run.
        A link is counted if its absolute link_value exceeds the threshold.

        Args:
            df: DataFrame with columns 'gene_i', 'gene_j', 'link_value', 'run'.
            threshold: Minimum absolute link_value to consider a link present (default 0.0).

        Returns:
            DataFrame with columns 'run', 'num_links', 'num_nodes', 'density_simple', 'density'.
        """
        self.logger.debug("Computing network density per run")

        # Filter links by threshold and ensure unique links per run
        df_filtered = df[df['link_value'].abs() > threshold].drop_duplicates(subset=['gene_i', 'gene_j', 'run'])

        # Group by run
        grouped = df_filtered.groupby('run')

        # Compute metrics per run
        results = []
        for run, group in grouped:
            # Number of links (unique edges)
            num_links = len(group)

            # Unique nodes (union of gene_i and gene_j)
            nodes = set(group['gene_i']).union(group['gene_j'])
            num_nodes = len(nodes)

            # Simple density: links / nodes
            density_simple = num_links / num_nodes if num_nodes > 0 else 0.0

            # Standard density: links / possible edges (directed graph)
            # Possible edges = N * (N - 1) for directed graphs
            possible_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
            density = num_links / possible_edges if possible_edges > 0 else 0.0

            results.append({
                'run': run,
                'num_links': num_links,
                'num_nodes': num_nodes,
                'density_simple': density_simple,
                'density': density
            })

        # Convert to DataFrame
        result_df = pd.DataFrame(results)
        self.logger.info(f"Computed network density for {len(result_df)} runs")
        return result_df
