from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, TypeVar, Union
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from bootstrap.utils import NetworkUtils

try:
    from config import AnalysisConfig
except ImportError:
    # Fallback if config module not available
    @dataclass
    class AnalysisConfig:
        total_runs: int = 64
        inner_group_size: int = 8
        support_threshold: float = 0.8
        fdr_threshold: float = 0.05
        epsilon: float = 1e-10

@dataclass
class NetworkData:
    """Data class to hold network analysis data.

    Attributes:
        Y: Input data matrix
        names: Node names
        N: Number of nodes
        M: Number of measurements/samples
    """
    Y: np.ndarray
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
        orig_index: Original index (int or list/array if multiple params)
        accumulated: Accumulated statistics (array or list of arrays)
        binned_freq: Binned frequencies (array or list of arrays)
        fp_rate: False positive rate at crossing (float or list/array)
        support: Support at crossing (float or list/array)
    """
    xnet: np.ndarray
    ssum: np.ndarray
    min_ab: np.ndarray
    sxnet: np.ndarray
    orig_index: Union[int, np.ndarray, List[int]]
    accumulated: Union[np.ndarray, List[np.ndarray]]
    binned_freq: Union[np.ndarray, List[np.ndarray]]
    fp_rate: Union[float, np.ndarray, List[float]]
    support: Union[float, np.ndarray, List[float]]
    gene_i: Optional[List[str]] = None
    gene_j: Optional[List[str]] = None

class Nestboot:
    """Class for performing Network Bootstrap False Discovery Rate analysis.
    
    This class implements the NB-FDR algorithm for network inference with 
    bootstrap-based confidence estimation.
    """

    def __init__(self, param: Optional[Union[logging.Logger, NetworkData, AnalysisConfig, dict]] = None) -> None:
        """Initialize Nestboot analyzer.

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
            df: DataFrame with columns: gene_i, gene_j, run, link_value, param_idx
            total_runs: Number of bootstrap runs
            inner_group_size: Size of groups for sign computation

        Returns:
            DataFrame with computed metrics
        """
        self.logger.debug("Computing assignment fractions")
        
        # Extract run numbers and filter out runs greater than or equal to total_runs
        df = df[df['run'].astype(int) < total_runs]
        
        # Check if param_idx exists, otherwise add it (for backward compatibility)
        if 'param_idx' not in df.columns:
            df['param_idx'] = 0
            
        # Group by param_idx and gene pairs 
        grouped = df.groupby(['param_idx', 'gene_i', 'gene_j'], as_index=False)
        run_counts = grouped.agg({
            'run': 'nunique',
            'link_value': 'sum'  # Sum of weights (we will divide by total_runs for average)
        })
        run_counts['Afrac'] = run_counts['run'] / total_runs
        # Average weight = Sum of weights / Total Runs (treating absent as 0)
        # This gives a "Bagging" style estimator
        run_counts['Aavg'] = run_counts['link_value'] / total_runs
        
        # Use drop to remove the temporary 'run' count and 'link_value' sum column if not needed
        # We keep Aavg
        results = run_counts.drop(columns=['run', 'link_value'])
        
        # Compute sign fractions for all links
        # We calculate the fraction of *present* runs where the link is positive.
        # This replaces the complex pivot/group logic which only worked for full support.
        
        self.logger.debug("Computing sign consistency")
        
        # Create a view/copy to avoid SettingWithCopy warning on original df
        sign_df = df[['param_idx', 'gene_i', 'gene_j', 'link_value']].copy()
        sign_df['is_pos'] = (sign_df['link_value'] > 0).astype(float)
        
        sign_stats = sign_df.groupby(['param_idx', 'gene_i', 'gene_j'])['is_pos'].mean()
        sign_stats.name = 'Asign_frac'
        
        results = results.merge(sign_stats, on=['param_idx', 'gene_i', 'gene_j'], how='left')
        results['Asign_frac'] = results['Asign_frac'].fillna(0.5) # Default to ambiguous if missing (shouldn't happen)
        
        return results

    def nb_fdr(
        self,
        normal_df: pd.DataFrame,
        shuffled_df: pd.DataFrame,
        init: int,
        data_dir: Path,
        fdr: float,
        boot: int,
        node_names: Optional[List[str]] = None
    ) -> NetworkResults:
        """Perform Network Bootstrap FDR analysis.

        Args:
            normal_df: Normal network data with gene_i, gene_j, run, link_value columns
            shuffled_df: Shuffled network data with same columns
            init: Number of initialization iterations
            data_dir: Directory for output files
            fdr: False Discovery Rate threshold
            boot: Number of bootstrap iterations
            node_names: List of node names to reconstruct the network matrix

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
                'Asign_frac': f'Asign_frac{suffix}',
                'Aavg': f'Aavg{suffix}'
            }, inplace=True)
        
        # Merge and compute metrics
        # Merging on param_idx too
        merged = pd.merge(
            agg_normal, agg_shuffled,
            on=['param_idx', 'gene_i', 'gene_j'],
            how='outer'
        ).fillna(0)
        
        # Pass FDR threshold to compute logic for dynamic thresholding
        results = self._compute_network_metrics(merged, fdr, node_names)
        
        self.logger.info("NB-FDR analysis completed successfully")
        return results

    def _compute_network_metrics(
        self, 
        merged: pd.DataFrame,
        target_fdr: float,
        node_names: Optional[List[str]] = None
    ) -> NetworkResults:
        """Compute network comparison metrics.

        Args:
            merged: Merged normal and shuffled network data
            target_fdr: Target False Discovery Rate (replacing fixed support_threshold)
            node_names: List of node names for matrix reconstruction

        Returns:
            NetworkResults object
        """
        eps = 1e-6  # Small value to prevent division by zero
        
        # Determine unique parameters
        if 'param_idx' in merged.columns:
            unique_params = sorted(merged['param_idx'].unique())
        else:
            unique_params = [0]
            merged['param_idx'] = 0
            
        n_params = len(unique_params)
        
        # Prepare lists to collect results per parameter
        xnet_list = []
        ssum_list = []
        min_ab_list = []
        sxnet_list = []
        accumulated_list = []
        binned_freq_list = []
        fp_rate_list = []
        support_list = []
        orig_index_list = []
        
        # We should iterate over range(max(unique_params) + 1) to be safe and fill with empty if missing
        if unique_params:
            try:
                max_param = int(max(unique_params))
                full_param_range = range(max_param + 1)
            except ValueError:
                full_param_range = [0]
        else:
            full_param_range = [0]
            
        for p in full_param_range:
            # Filter for current parameter
            sub_merged = merged[merged['param_idx'] == p].copy()
            
            # --- Dynamic FDR Threshold Calculation ---
            if sub_merged.empty:
                best_t = 0.8 # Fallback
                current_fdr = 0.0
            else:
                # Find smallest t such that FDR(t) <= target_fdr
                # FDR(t) = (Shuf_counts >= t) / (Norm_counts >= t)
                # Scan t from 0.05 to 1.0
                t_vals = np.linspace(0.05, 1.0, 96)
                
                # Pre-fetch arrays
                n_norm_arr = sub_merged['Afrac_norm'].values
                n_shuf_arr = sub_merged['Afrac_shuf'].values
                
                # Default to highest if none satisfy
                best_t = 1.0 
                found = False
                
                # Check from high to low to find the Cutoff
                # Usually we want the *lowest* t that is still valid (Maximum Recall)
                # But we must ensure all t' > t are also valid? Not necessarily monotonic in practice but roughly.
                # Standard Approach: Find the lowest t with FDR < target.
                
                for t in reversed(t_vals):
                    n_norm = (n_norm_arr >= t).sum()
                    if n_norm == 0:
                        continue
                    n_shuf = (n_shuf_arr >= t).sum()
                    
                    fdr_est = n_shuf / n_norm
                    
                    if fdr_est <= target_fdr:
                        best_t = t
                    else:
                        # FDR exceeded, stop going lower. The previous t (larger) was the limit.
                        break
            
            support_threshold = best_t
            
            if sub_merged.empty:
                # Handle empty case (no links found for this parameter)
                if node_names:
                    N = len(node_names)
                    xnet = np.zeros((N, N))
                    ssum = np.zeros((N, N))
                    min_ab = np.zeros((N, N))
                    sxnet = np.zeros((N, N))
                else:
                    xnet = np.array([[0]])
                    ssum = np.array([[0]])
                    min_ab = np.array([[0]])
                    sxnet = np.array([[0]])
                
                accumulated = np.zeros((1, 2))
                binned_freq = np.zeros(10)
                fp = 0.0
                curr_orig_index = 0
            else:
                # Compute metrics vectors
                xnet_vec = (sub_merged['Afrac_norm'] >= support_threshold).astype(float)
                
                # Sign determination: Map [0,1] back to {-1, +1}
                # Asign_frac_norm is freq of positive signs.
                # >0.5 is (+), <0.5 is (-).
                # Handle exact 0.5 (ambiguous) -> 1 or 0? 
                # Let's align with existing dominance.
                signs = np.sign(sub_merged['Asign_frac_norm'] - 0.5)
                signs[signs == 0] = 1 # Force non-zero
                
                ssum_vec = signs
                min_ab_vec = sub_merged['Afrac_norm']
                
                # sxnet: Continuous Ranking Score
                # Use Frequency (Afrac) * Sign. 
                # This provides [-1, 1] ranking.
                sxnet_vec = min_ab_vec * ssum_vec
                
                # Reconstruct matrices if node_names provided
                if node_names:
                    N = len(node_names)
                    xnet = np.zeros((N, N))
                    ssum = np.zeros((N, N))
                    min_ab = np.zeros((N, N))
                    sxnet = np.zeros((N, N))
                    
                    # Map names to indices
                    name_to_idx = {name: i for i, name in enumerate(node_names)}
                    
                    rows = sub_merged['gene_i'].map(name_to_idx)
                    cols = sub_merged['gene_j'].map(name_to_idx)
                    
                    # Drop unmapped
                    valid = rows.notna() & cols.notna()
                    if not valid.all():
                        self.logger.warning("Some genes in results not found in node_names")
                    
                    r = rows[valid].astype(int).values
                    c = cols[valid].astype(int).values
                    
                    # Update vectors to match valid
                    valid_xnet = xnet_vec[valid].values
                    valid_ssum = ssum_vec[valid].values
                    valid_min_ab = min_ab_vec[valid].values
                    valid_sxnet = sxnet_vec[valid].values
                    
                    xnet[r, c] = valid_xnet
                    ssum[r, c] = valid_ssum
                    min_ab[r, c] = valid_min_ab
                    sxnet[r, c] = valid_sxnet
                else:
                    xnet = xnet_vec.values
                    ssum = ssum_vec.values
                    min_ab = min_ab_vec.values
                    sxnet = sxnet_vec.values
                
            xnet_list.append(xnet)
            ssum_list.append(ssum)
            min_ab_list.append(min_ab)
            sxnet_list.append(sxnet)
            
            # Additional metrics
            accumulated = self._compute_accumulated_stats(sub_merged)
            binned_freq = self._compute_binned_frequencies(sub_merged)
            
            accumulated_list.append(accumulated)
            binned_freq_list.append(binned_freq)
            fp_rate_list.append(0.0) # Placeholder/Depr
            support_list.append(support_threshold)
            orig_index_list.append(int(support_threshold * 100))

        # Stack results
        if n_params == 1:
            return NetworkResults(
                xnet=xnet_list[0],
                ssum=ssum_list[0],
                min_ab=min_ab_list[0],
                sxnet=sxnet_list[0],
                orig_index=orig_index_list[0],
                accumulated=accumulated_list[0],
                binned_freq=binned_freq_list[0],
                fp_rate=fp_rate_list[0],
                support=support_list[0],
                gene_i=merged['gene_i'].tolist(),
                gene_j=merged['gene_j'].tolist()
            )
        else:
            # Stack arrays along 3rd dimension: (N, N, n_params)
            # Assuming xnet_list contains (N, N) arrays
            if node_names:
                xnet_stack = np.stack(xnet_list, axis=2)
                ssum_stack = np.stack(ssum_list, axis=2)
                min_ab_stack = np.stack(min_ab_list, axis=2)
                sxnet_stack = np.stack(sxnet_list, axis=2)
            else:
                # If not using node names, we can't easily stack if shapes differ
                # But typically they would share same gene set from merged
                # For now, let's assume node_names is always provided via run_nestboot
                # or return list of arrays?
                # Stacking 1D arrays -> (L, n_params) ? No, lengths might differ.
                # Just return lists if no node_names?
                # But run_nestboot ensures node_names is mostly present.
                xnet_stack = xnet_list # Fallback
                ssum_stack = ssum_list
                min_ab_stack = min_ab_list
                sxnet_stack = sxnet_list

            return NetworkResults(
                xnet=xnet_stack,
                ssum=ssum_stack,
                min_ab=min_ab_stack,
                sxnet=sxnet_stack,
                orig_index=orig_index_list,
                accumulated=accumulated_list,
                binned_freq=binned_freq_list,
                fp_rate=fp_rate_list,
                support=support_list,
                gene_i=merged['gene_i'].tolist(),
                gene_j=merged['gene_j'].tolist()
            )

    def _accumulate(
        self,
        boo_alink: List[np.ndarray],
        boo_shuffle_alink: List[np.ndarray],
        init: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
               np.ndarray, np.ndarray, np.ndarray]:
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
        
        estimated_support_net: List[np.ndarray] = []
        estimated_shuffle_net: List[np.ndarray] = []
        overlaps_support: List[np.ndarray] = []
        overlaps_shuffle: List[np.ndarray] = []

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
        boot_links: List[np.ndarray],
        idx: int,
        estimated_net: List[np.ndarray],
        init: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        overlaps: List[np.ndarray],
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

    def _find_fdr_cutoff(
        self,
        binned_freq: np.ndarray,
        accumulated: np.ndarray,
        fdr: float
    ) -> int:
        """Find the index where FDR threshold is crossed.

        Args:
            binned_freq: Binned frequency data
            accumulated: Accumulated statistics
            fdr: False discovery rate threshold

        Returns:
            Index where FDR threshold is crossed
        """
        # Simple implementation: find first index where FP rate <= FDR
        for i in range(len(binned_freq)):
            if i < len(accumulated):
                fp_rate = accumulated[i, 1] / accumulated[i, 0] if accumulated[i, 0] > 0 else 0.0
                if fp_rate <= fdr:
                    return i
        return len(binned_freq) - 1  # Return last index if no cutoff found

    def _get_plottable_results(
        self,
        freq: np.ndarray,
        init: int,
        accumulated: np.ndarray,
        overlaps_support: np.ndarray,
        fdr: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
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

    def _compute_accumulated_stats(self, merged: pd.DataFrame) -> np.ndarray:
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
    
    def _compute_binned_frequencies(self, merged: pd.DataFrame, bins: int = 10) -> np.ndarray:
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

    def run_nestboot(
        self,
        dataset: Any,
        inference_method: Any,
        nest_runs: int = 50,
        boot_runs: int = 50,
        seed: int = 42,
        method_kwargs: Optional[Dict[str, Any]] = None,
        method_params: Optional[Dict[str, Any]] = None
    ) -> NetworkResults:
        """Run NestBoot analysis with bootstrapping and network inference.
        
        Args:
            dataset: Data object containing the dataset
            inference_method: Inference method - can be:
                - A callable function that takes (dataset, **kwargs)
                - An inference method class (like Lasso) that will be called with method_params
            nest_runs: Number of outer runs
            boot_runs: Number of inner runs
            seed: Random seed
            method_kwargs: Additional arguments for callable inference_method
            method_params: Parameters to pass to inference method class (e.g., {'alpha_range': zetavec})
            
        Returns:
            NetworkResults object
        """
        import copy
        from analyze.Data import Data
        
        if method_kwargs is None:
            method_kwargs = {}
        if method_params is None:
            method_params = {}
            
        np.random.seed(seed)
        
        bootstrap_data = []
        shuffled_data = []
        
        # Access underlying Dataset object
        if hasattr(dataset, 'Y') and dataset.Y is not None:
            ds_obj = dataset
        elif hasattr(dataset, 'data') and dataset.data is not None:
            ds_obj = dataset.data
        else:
            ds_obj = dataset
            
        n_genes = ds_obj.N
        n_samples = ds_obj.M
        
        gene_names_list = ds_obj.gene_names if hasattr(ds_obj, 'gene_names') and ds_obj.gene_names else [f"Gene_{i:02d}" for i in range(n_genes)]
        
        # Keep clean copies of the original data for bootstrapping
        original_Y = ds_obj.Y.copy()
        original_P = ds_obj.P.copy()
        
        for outer_run in range(nest_runs):
            self.logger.info(f"NestBoot outer run {outer_run + 1}/{nest_runs}")
            
            for boot_run in range(boot_runs):
                try:
                    # Bootstrap
                    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    
                    # Create a new Dataset object with bootstrapped data
                    from datastruct.Dataset import Dataset
                    bootstrap_dataset_obj = Dataset()
                    bootstrap_dataset_obj._Y = original_Y[:, bootstrap_indices]
                    bootstrap_dataset_obj._P = original_P[:, bootstrap_indices]
                    bootstrap_dataset_obj._network = getattr(ds_obj, '_network', None)
                    bootstrap_dataset_obj._names = getattr(ds_obj, '_names', None)
                    bootstrap_dataset_obj._E = getattr(ds_obj, '_E', None)
                    bootstrap_dataset_obj._lambda = getattr(ds_obj, '_lambda', None)
                    bootstrap_dataset_obj._dataset_name = getattr(ds_obj, '_dataset_name', "Bootstrap")
                    bootstrap_dataset = Data(bootstrap_dataset_obj)
                    
                    # Run inference
                    if callable(inference_method) and method_params:
                        # Method is a class/function, call it with method_params
                        network_result = inference_method(bootstrap_dataset, **method_params)
                    else:
                        # Method is a callable function, call it with method_kwargs
                        network_result = inference_method(bootstrap_dataset, **method_kwargs)
                    
                    # Handle different return types (tuple or just network)
                    if isinstance(network_result, tuple):
                        network_matrix = network_result[0]
                    else:
                        network_matrix = network_result
                        
                    # Handle 3D array or 2D array and extract links
                    param_list = []
                    
                    if hasattr(network_matrix, 'ndim') and network_matrix.ndim == 3:
                        # 3D array: (genes, genes, params)
                        n_params = network_matrix.shape[2]
                        for p in range(n_params):
                            mat = network_matrix[:, :, p]
                            rows, cols = np.where(np.abs(mat) > 1e-6)
                            # Remove diagonal
                            mask = rows != cols
                            rows, cols = rows[mask], cols[mask]
                            vals = mat[rows, cols]
                            
                            for r, c, v in zip(rows, cols, vals):
                                bootstrap_data.append({
                                    'gene_i': gene_names_list[r],
                                    'gene_j': gene_names_list[c],
                                    'run': outer_run * boot_runs + boot_run, # Unique run ID
                                    'link_value': v,
                                    'param_idx': p
                                })
                    else:
                        # 2D case
                         mat = network_matrix
                         rows, cols = np.where(np.abs(mat) > 1e-6)
                         mask = rows != cols
                         rows, cols = rows[mask], cols[mask]
                         vals = mat[rows, cols]
                         
                         for r, c, v in zip(rows, cols, vals):
                             bootstrap_data.append({
                                 'gene_i': gene_names_list[r],
                                 'gene_j': gene_names_list[c],
                                 'run': outer_run * boot_runs + boot_run, # Unique run ID
                                 'link_value': v,
                                 'param_idx': 0
                             })
                    
                    # Shuffle
                    # CRITICAL FIX: We must break the relationship between Y and P.
                    # Previously, we shuffled both with the same indices, which just reordered the samples
                    # but preserved the Y-P correspondence, leading to the "shuffled" data containing the real network signal.
                    
                    shuffle_indices_ya = np.random.permutation(n_samples)
                    shuffle_indices_yb = np.random.permutation(n_genes)
                    
                    # We can keep P in original order, or shuffle it independently. 
                    # Shuffling Y against fixed P is sufficient to break links.
                    # If we really want to be random, we can shuffle P independently too, 
                    # but keeping one fixed serves the purpose of breaking the pair (Y_i, P_i).
                    
                    shuffled_dataset_obj = Dataset()
                    # Apply both column (sample) and row (gene) shuffling
                    # Use a temporary variable to ensure both are applied
                    Y_shuf = original_Y.copy()
                    Y_shuf = Y_shuf[:, shuffle_indices_ya] # Shuffle samples
                    Y_shuf = Y_shuf[shuffle_indices_yb, :] # Shuffle genes
                    shuffled_dataset_obj._Y = Y_shuf
                    shuffled_dataset_obj._P = original_P # Keep P as is (or original_P.copy() if safety needed)
                    # Note: We must ensure P has same dimensions. original_P is (genes, samples).
                    
                    shuffled_dataset_obj._network = ds_obj._network
                    shuffled_dataset_obj._names = ds_obj._names
                    shuffled_dataset_obj._E = ds_obj._E
                    shuffled_dataset_obj._lambda = ds_obj._lambda
                    shuffled_dataset_obj._dataset_name = ds_obj._dataset_name
                    shuffled_dataset = Data(shuffled_dataset_obj)
                    
                    # Run inference on shuffled data
                    if callable(inference_method) and method_params:
                        # Method is a class/function, call it with method_params
                        shuffled_result = inference_method(shuffled_dataset, **method_params)
                    else:
                        # Method is a callable function, call it with method_kwargs
                        shuffled_result = inference_method(shuffled_dataset, **method_kwargs)
                        
                    if isinstance(shuffled_result, tuple):
                        shuffled_network = shuffled_result[0]
                    else:
                        shuffled_network = shuffled_result
                    
                    # Store shuffled links
                    if hasattr(shuffled_network, 'ndim') and shuffled_network.ndim == 3:
                        n_params = shuffled_network.shape[2]
                        for p in range(n_params):
                            mat = shuffled_network[:, :, p]
                            rows, cols = np.where(np.abs(mat) > 1e-6)
                            mask = rows != cols
                            rows, cols = rows[mask], cols[mask]
                            vals = mat[rows, cols]
                            for r, c, v in zip(rows, cols, vals):
                                shuffled_data.append({
                                    'gene_i': gene_names_list[r],
                                    'gene_j': gene_names_list[c],
                                    'run': outer_run * boot_runs + boot_run, # Unique run ID
                                    'link_value': v, 
                                    'param_idx': p
                                })
                    else:
                        mat = shuffled_network
                        rows, cols = np.where(np.abs(mat) > 1e-6)
                        mask = rows != cols
                        rows, cols = rows[mask], cols[mask]
                        vals = mat[rows, cols]
                        for r, c, v in zip(rows, cols, vals):
                             shuffled_data.append({
                                 'gene_i': gene_names_list[r],
                                 'gene_j': gene_names_list[c],
                                 'run': outer_run * boot_runs + boot_run, # Unique run ID
                                 'link_value': v,
                                 'param_idx': 0
                             })
                                
                except Exception as e:
                    self.logger.error(f"Bootstrap iteration failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        # Convert to DataFrames
        normal_df = pd.DataFrame(bootstrap_data)
        shuffled_df = pd.DataFrame(shuffled_data)
        
        if len(normal_df) == 0:
            raise ValueError("No bootstrap data generated")
            
        # Run NB-FDR analysis
        return self.nb_fdr(
            normal_df=normal_df,
            shuffled_df=shuffled_df,
            init=nest_runs * boot_runs, # Total runs is now product of nested loops
            data_dir=Path("."),
            fdr=self.config.fdr_threshold if hasattr(self, 'config') else 0.05,
            boot=boot_runs,
            node_names=gene_names_list
        )
