"""
Strategy pattern implementations for different analysis methods.

This module provides concrete strategy implementations for various network
analysis approaches using the Strategy pattern.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import logging

from ..core.base import AbstractAnalysisStrategy, AbstractAnalysisResult, AnalysisConfig


class NBFDRAnalysisResult(AbstractAnalysisResult):
    """Result class for NB-FDR analysis."""
    
    def __init__(self, xnet: np.ndarray, ssum: np.ndarray, min_ab: np.ndarray,
                 sxnet: np.ndarray, orig_index: int, accumulated: np.ndarray,
                 binned_freq: np.ndarray, fp_rate: float, support: float):
        super().__init__()
        self._metrics = {
            'xnet': xnet,
            'ssum': ssum,
            'min_ab': min_ab,
            'sxnet': sxnet,
            'orig_index': orig_index,
            'accumulated': accumulated,
            'binned_freq': binned_freq,
            'fp_rate': fp_rate,
            'support': support
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        return {
            'network_size': len(self._metrics['xnet']),
            'false_positive_rate': self._metrics['fp_rate'],
            'support_threshold': self._metrics['support'],
            'original_index': self._metrics['orig_index'],
            'num_significant_links': int(np.sum(self._metrics['xnet'] > 0))
        }
    
    def validate_results(self) -> bool:
        """Validate result consistency."""
        try:
            # Check if all arrays have compatible shapes
            xnet = self._metrics['xnet']
            ssum = self._metrics['ssum']
            min_ab = self._metrics['min_ab']
            sxnet = self._metrics['sxnet']
            
            if not (len(xnet) == len(ssum) == len(min_ab) == len(sxnet)):
                return False
            
            # Check if FP rate is reasonable
            fp_rate = self._metrics['fp_rate']
            if not (0 <= fp_rate <= 1):
                return False
            
            # Check if support is reasonable
            support = self._metrics['support']
            if not (0 <= support <= 1):
                return False
            
            return True
        except Exception:
            return False


class NBFDRStrategy(AbstractAnalysisStrategy[NBFDRAnalysisResult]):
    """Strategy for Network Bootstrap FDR analysis."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def execute(self, normal_df: pd.DataFrame, shuffled_df: pd.DataFrame) -> NBFDRAnalysisResult:
        """Execute NB-FDR analysis strategy."""
        if not self.validate_inputs(normal_df, shuffled_df):
            raise ValueError("Invalid input data")
        
        self.logger.info("Starting NB-FDR analysis")
        
        # Compute assignment fractions
        agg_normal = self._compute_assign_frac(normal_df)
        agg_shuffled = self._compute_assign_frac(shuffled_df)
        
        # Rename columns for merging
        agg_normal.rename(columns={
            'Afrac': 'Afrac_norm',
            'Asign_frac': 'Asign_frac_norm'
        }, inplace=True)
        agg_shuffled.rename(columns={
            'Afrac': 'Afrac_shuf',
            'Asign_frac': 'Asign_frac_shuf'
        }, inplace=True)
        
        # Merge and compute metrics
        merged = pd.merge(agg_normal, agg_shuffled, on=['gene_i', 'gene_j'])
        
        # Compute network metrics
        results = self._compute_network_metrics(merged)
        
        self.logger.info("NB-FDR analysis completed")
        return results
    
    def validate_inputs(self, normal_df: pd.DataFrame, shuffled_df: pd.DataFrame) -> bool:
        """Validate input DataFrames."""
        required_columns = ['gene_i', 'gene_j', 'run', 'link_value']
        
        for df, name in [(normal_df, 'normal'), (shuffled_df, 'shuffled')]:
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in {name} DataFrame")
                return False
            
            if df.empty:
                self.logger.error(f"{name} DataFrame is empty")
                return False
        
        return True
    
    def _compute_assign_frac(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute assignment fractions for network links."""
        # Filter runs
        df = df[df['run'].astype(str).str.extract(r'(\d+)')[0].astype(int) < self.config.total_runs]
        
        # Group by gene pairs
        grouped = df.groupby(['gene_i', 'gene_j'], as_index=False)
        run_counts = grouped['run'].nunique()
        run_counts['Afrac'] = run_counts['run'] / self.config.total_runs
        results = run_counts.drop(columns=['run'])
        
        # Compute sign fractions for full support links
        full_support = results[results['Afrac'] >= 1]
        sign_fracs = {}
        
        for _, row in full_support.iterrows():
            gene_i, gene_j = row['gene_i'], row['gene_j']
            group = df[(df['gene_i'] == gene_i) & (df['gene_j'] == gene_j)]
            
            run_values = (group.drop_duplicates('run')
                         .sort_values('run')['link_value']
                         .values[:self.config.total_runs])
            
            num_groups = len(run_values) // self.config.inner_group_size
            if num_groups > 0:
                r = run_values[:num_groups * self.config.inner_group_size]
                r = r.reshape(num_groups, self.config.inner_group_size)
                pos_frac = (r > 0).mean(axis=1).mean()
                sign = 2 * pos_frac - 1
            else:
                sign = 0
            
            sign_fracs[(gene_i, gene_j)] = sign
        
        results['Asign_frac'] = results.apply(
            lambda row: sign_fracs.get((row['gene_i'], row['gene_j']), 0), axis=1
        )
        
        return results
    
    def _compute_network_metrics(self, merged: pd.DataFrame) -> NBFDRAnalysisResult:
        """Compute network comparison metrics."""
        eps = self.config.epsilon
        threshold = self.config.support_threshold
        
        # Compute basic metrics
        xnet = (merged['Afrac_norm'] >= threshold).astype(float)
        ssum = np.sign(merged['Asign_frac_norm'])
        min_ab = merged['Afrac_norm']
        sxnet = xnet * ssum
        
        # Compute additional metrics
        ff = merged['Afrac_norm'] - merged['Afrac_shuf']
        fp = merged['Afrac_shuf'] / (merged['Afrac_norm'] + eps)
        
        # Compute accumulated statistics
        accumulated = self._compute_accumulated_stats(merged)
        binned_freq = self._compute_binned_frequencies(merged)
        
        return NBFDRAnalysisResult(
            xnet=xnet.values,
            ssum=ssum.values,
            min_ab=min_ab.values,
            sxnet=sxnet.values,
            orig_index=int(threshold * 100),
            accumulated=accumulated,
            binned_freq=binned_freq,
            fp_rate=fp.mean(),
            support=threshold
        )
    
    def _compute_accumulated_stats(self, merged: pd.DataFrame) -> np.ndarray:
        """Compute accumulated statistics."""
        # Simplified implementation - in practice this would be more complex
        norm_vals = merged['Afrac_norm'].values
        shuf_vals = merged['Afrac_shuf'].values
        
        bins = np.linspace(0, 1, 11)
        norm_hist, _ = np.histogram(norm_vals, bins=bins)
        shuf_hist, _ = np.histogram(shuf_vals, bins=bins)
        
        # Normalize
        norm_hist = norm_hist / norm_hist.sum() if norm_hist.sum() > 0 else norm_hist
        shuf_hist = shuf_hist / shuf_hist.sum() if shuf_hist.sum() > 0 else shuf_hist
        
        return np.column_stack([norm_hist, shuf_hist])
    
    def _compute_binned_frequencies(self, merged: pd.DataFrame, bins: int = 10) -> np.ndarray:
        """Compute binned frequency statistics."""
        norm_vals = merged['Afrac_norm'].values
        shuf_vals = merged['Afrac_shuf'].values
        
        bin_edges = np.linspace(0, 1, bins + 1)
        norm_hist, _ = np.histogram(norm_vals, bins=bin_edges)
        shuf_hist, _ = np.histogram(shuf_vals, bins=bin_edges)
        
        # Normalize
        total_norm = norm_hist.sum()
        total_shuf = shuf_hist.sum()
        
        if total_norm > 0:
            norm_hist = norm_hist / total_norm
        if total_shuf > 0:
            shuf_hist = shuf_hist / total_shuf
        
        return np.column_stack([norm_hist, shuf_hist]).flatten()


class BootstrapSamplingStrategy(AbstractAnalysisStrategy[pd.DataFrame]):
    """Strategy for bootstrap sampling of network data."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def execute(self, data: pd.DataFrame, n_samples: int = None) -> pd.DataFrame:
        """Execute bootstrap sampling strategy."""
        if not self.validate_inputs(data):
            raise ValueError("Invalid input data")
        
        n_samples = n_samples or self.config.total_runs
        
        self.logger.info(f"Generating {n_samples} bootstrap samples")
        
        bootstrap_samples = []
        for i in range(n_samples):
            sample = self._generate_bootstrap_sample(data, i)
            bootstrap_samples.append(sample)
        
        result = pd.concat(bootstrap_samples, ignore_index=True)
        
        self.logger.info("Bootstrap sampling completed")
        return result
    
    def validate_inputs(self, data: pd.DataFrame) -> bool:
        """Validate input DataFrame."""
        if data.empty:
            self.logger.error("Input DataFrame is empty")
            return False
        
        return True
    
    def _generate_bootstrap_sample(self, data: pd.DataFrame, sample_id: int) -> pd.DataFrame:
        """Generate a single bootstrap sample."""
        # Bootstrap sample with replacement
        n_samples = len(data)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        sample = data.iloc[bootstrap_indices].copy()
        sample['bootstrap_run'] = sample_id
        
        return sample


class NetworkComparisonStrategy(AbstractAnalysisStrategy[Dict[str, Any]]):
    """Strategy for comparing multiple networks."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def execute(self, networks: List[np.ndarray], names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute network comparison strategy."""
        if not self.validate_inputs(networks):
            raise ValueError("Invalid input networks")
        
        names = names or [f"network_{i}" for i in range(len(networks))]
        
        self.logger.info(f"Comparing {len(networks)} networks")
        
        # Compute pairwise similarities
        similarities = self._compute_pairwise_similarities(networks)
        
        # Compute network properties
        properties = self._compute_network_properties(networks, names)
        
        # Compute consensus network
        consensus = self._compute_consensus_network(networks)
        
        result = {
            'similarities': similarities,
            'properties': properties,
            'consensus_network': consensus,
            'network_names': names
        }
        
        self.logger.info("Network comparison completed")
        return result
    
    def validate_inputs(self, networks: List[np.ndarray]) -> bool:
        """Validate input networks."""
        if not networks:
            self.logger.error("No networks provided")
            return False
        
        # Check if all networks have the same shape
        first_shape = networks[0].shape
        for i, network in enumerate(networks[1:], 1):
            if network.shape != first_shape:
                self.logger.error(f"Network {i} has different shape than first network")
                return False
        
        return True
    
    def _compute_pairwise_similarities(self, networks: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise similarities between networks."""
        n_networks = len(networks)
        similarities = np.zeros((n_networks, n_networks))
        
        for i in range(n_networks):
            for j in range(i, n_networks):
                if i == j:
                    similarities[i, j] = 1.0
                else:
                    # Compute Jaccard similarity for binary networks
                    net_i = (networks[i] != 0).astype(int)
                    net_j = (networks[j] != 0).astype(int)
                    
                    intersection = np.sum(net_i * net_j)
                    union = np.sum((net_i + net_j) > 0)
                    
                    similarity = intersection / union if union > 0 else 0.0
                    similarities[i, j] = similarity
                    similarities[j, i] = similarity
        
        return similarities
    
    def _compute_network_properties(self, networks: List[np.ndarray], names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Compute properties for each network."""
        properties = {}
        
        for network, name in zip(networks, names):
            n_nodes = network.shape[0]
            n_edges = np.sum(network != 0)
            density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
            
            properties[name] = {
                'n_nodes': n_nodes,
                'n_edges': int(n_edges),
                'density': density,
                'is_symmetric': np.allclose(network, network.T),
                'has_self_loops': np.any(np.diag(network) != 0)
            }
        
        return properties
    
    def _compute_consensus_network(self, networks: List[np.ndarray]) -> np.ndarray:
        """Compute consensus network from multiple networks."""
        if not networks:
            return np.array([])
        
        # Stack networks and compute mean
        stacked = np.stack(networks, axis=0)
        consensus = np.mean(stacked, axis=0)
        
        # Apply threshold to create binary consensus
        threshold = self.config.support_threshold
        binary_consensus = (consensus >= threshold).astype(float)
        
        return binary_consensus


class AnalysisStrategyFactory:
    """Factory for creating analysis strategies."""
    
    _strategies = {
        'nb_fdr': NBFDRStrategy,
        'bootstrap_sampling': BootstrapSamplingStrategy,
        'network_comparison': NetworkComparisonStrategy
    }
    
    @classmethod
    def create_strategy(cls, strategy_name: str, config: AnalysisConfig) -> AbstractAnalysisStrategy:
        """Create a strategy instance."""
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type) -> None:
        """Register a new strategy."""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies."""
        return list(cls._strategies.keys())
