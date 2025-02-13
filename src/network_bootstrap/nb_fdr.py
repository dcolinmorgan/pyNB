from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, TypeVar, Union
import numpy as np
import numpy.typing as npt
import pandas as pd
import logging
from pathlib import Path
from .utils import NetworkUtils

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

    def __init__(self, param: Optional[Union[logging.Logger, NetworkData]] = None) -> None:
        """Initialize NetworkBootstrap analyzer.

        Args:
            param: Optional parameter which can be either a logger instance or a NetworkData instance.
                   If a logger is provided, it will be used. If a NetworkData object is provided,
                   it is stored as `self.data` and a default logger is created.
        """
        if param is None:
            self.logger = logging.getLogger(__name__)
            self.data = None
        elif isinstance(param, logging.Logger):
            self.logger = param
            self.data = None
        elif isinstance(param, NetworkData):
            self.data = param
            self.logger = logging.getLogger(__name__)
        else:
            raise TypeError("Invalid type for parameter. Expected logging.Logger or NetworkData.")
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
        method: str,
        init: int,
        data_dir: Path,
        fdr: float,
        boot: int
    ) -> NetworkResults:
        """Perform Network Bootstrap FDR analysis.

        Args:
            normal_df: Normal network data with gene_i, gene_j, run, link_value columns
            shuffled_df: Shuffled network data with same columns
            method: Analysis method name
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
        """Plot analysis results with dual y-axes.

        The x-axis represents support (Afrac_norm),
        left y-axis displays the average overlap between the normal
        and shuffled networks, and the right y-axis shows the link frequency.

        Args:
            merged: Merged DataFrame with 'Afrac_norm' and 'Afrac_shuf' columns.
            plot_file: Path to save the plot image.
            bins: Number of bins to use for support.
        """
        # Define bins for support using measured bootstrap support (Afrac_norm)
        support_bins = np.linspace(0, 1, bins + 1)
        bin_centers = (support_bins[:-1] + support_bins[1:]) / 2
        
        # Compute bin counts for measured (normal) and null (shuffled) data
        counts_norm, _ = np.histogram(merged['Afrac_norm'], bins=support_bins)
        counts_shuf, _ = np.histogram(merged['Afrac_shuf'], bins=support_bins)
        
        # For each bin, compute the support metric as:
        # support_metric = (counts_norm - counts_shuf) / counts_norm   (if counts_norm > 0, else 0)
        support_metric = []
        for i in range(len(counts_norm)):
            if counts_norm[i] > 0:
                support_metric.append((counts_norm[i] - counts_shuf[i]) / counts_norm[i])
            else:
                support_metric.append(0)
        support_metric = np.array(support_metric)
        
        import matplotlib.pyplot as plt
        fig, ax_left = plt.subplots(figsize=(10, 6))
        
        # Plot support metric (i.e. 1 - FDR) on the left y-axis
        ax_left.plot(bin_centers, support_metric, color='tab:blue', marker='o', label='Support Metric')
        ax_left.set_xlabel('Bootstrap Support (Afrac_norm)')
        ax_left.set_ylabel('Support Metric (1 - FDR per bin)', color='black')
        ax_left.tick_params(axis='y', labelcolor='black')
        
        # Create right y-axis for link frequencies from measured and null data
        ax_right = ax_left.twinx()
        # Use previously computed counts_norm and counts_shuf to get normalized frequencies
        freq_norm = counts_norm.astype(float) / counts_norm.sum() if counts_norm.sum() > 0 else counts_norm.astype(float)
        freq_shuf = counts_shuf.astype(float) / counts_shuf.sum() if counts_shuf.sum() > 0 else counts_shuf.astype(float)
        ax_right.plot(bin_centers, freq_norm, color='tab:red', marker='s', label='Freq Normal')
        ax_right.plot(bin_centers, freq_shuf, color='tab:green', marker='^', label='Freq Shuffled')
        ax_right.set_ylabel('Link Frequency', color='black')
        ax_right.tick_params(axis='y', labelcolor='black')
        
        # Combine legends from both axes
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax_right.legend(lines_left + lines_right, labels_left + labels_right, loc='upper right')
        
        fig.tight_layout()
        plt.title('Analysis Results: Overlap & Link Frequency vs Support')
        plt.savefig(plot_file, dpi=300)
        plt.close()

    # Additional helper methods would go here... 
