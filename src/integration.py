"""
Integration module that connects the existing bootstrap code with the new OOP architecture.

This module provides adapters and integration points to use the existing
NetworkBootstrap class with the new OOP patterns while maintaining
backward compatibility.
"""

from typing import Optional, Union, Any
import logging
from pathlib import Path

try:
    # Try to import the existing bootstrap code
    from .bootstrap.nb_fdr import NetworkBootstrap as LegacyNetworkBootstrap
    from .bootstrap.nb_fdr import NetworkData, NetworkResults
    LEGACY_AVAILABLE = True
except ImportError:
    # Fallback if legacy code not available
    LEGACY_AVAILABLE = False
    LegacyNetworkBootstrap = None
    NetworkData = None
    NetworkResults = None

try:
    # Import the new OOP components
    from .core.base import AnalysisConfig, AbstractAnalysisResult
    from .improved_bootstrap import ImprovedNetworkBootstrap, NetworkBootstrapFacade
    OOP_AVAILABLE = True
except ImportError:
    OOP_AVAILABLE = False
    AnalysisConfig = None
    AbstractAnalysisResult = None
    ImprovedNetworkBootstrap = None
    NetworkBootstrapFacade = None


class LegacyResultsAdapter(AbstractAnalysisResult):
    """Adapter to make legacy NetworkResults compatible with new interface."""
    
    def __init__(self, legacy_results: 'NetworkResults'):
        super().__init__()
        self.legacy_results = legacy_results
        self._metrics = {
            'xnet': legacy_results.xnet,
            'ssum': legacy_results.ssum,
            'min_ab': legacy_results.min_ab,
            'sxnet': legacy_results.sxnet,
            'orig_index': legacy_results.orig_index,
            'accumulated': legacy_results.accumulated,
            'binned_freq': legacy_results.binned_freq,
            'fp_rate': legacy_results.fp_rate,
            'support': legacy_results.support
        }
    
    def get_summary(self):
        """Get result summary."""
        return {
            'network_size': len(self.legacy_results.xnet),
            'false_positive_rate': self.legacy_results.fp_rate,
            'support_threshold': self.legacy_results.support,
            'original_index': self.legacy_results.orig_index,
            'num_significant_links': int(sum(self.legacy_results.xnet > 0))
        }
    
    def validate_results(self):
        """Validate result consistency."""
        try:
            # Basic validation
            return (self.legacy_results.xnet is not None and
                    len(self.legacy_results.xnet) > 0 and
                    0 <= self.legacy_results.fp_rate <= 1)
        except Exception:
            return False


class LegacyNetworkBootstrapAdapter:
    """Adapter that makes the legacy NetworkBootstrap compatible with new OOP interface."""
    
    def __init__(self, legacy_instance: Optional['LegacyNetworkBootstrap'] = None):
        if not LEGACY_AVAILABLE:
            raise ImportError("Legacy NetworkBootstrap not available")
        
        self.legacy_nb = legacy_instance or LegacyNetworkBootstrap()
        self.logger = logging.getLogger(__name__)
    
    def compute_assign_frac(self, df, total_runs=64, inner_group_size=8):
        """Delegate to legacy implementation."""
        return self.legacy_nb.compute_assign_frac(df, total_runs, inner_group_size)
    
    def nb_fdr(self, normal_df, shuffled_df, init, data_dir, fdr, boot):
        """Delegate to legacy implementation and adapt results."""
        legacy_results = self.legacy_nb.nb_fdr(normal_df, shuffled_df, init, data_dir, fdr, boot)
        return LegacyResultsAdapter(legacy_results)
    
    def export_results(self, results, txt_file):
        """Export results using legacy method."""
        if isinstance(results, LegacyResultsAdapter):
            self.legacy_nb.export_results(results.legacy_results, txt_file)
        else:
            # Try to convert to legacy format
            self.logger.warning("Converting new format results to legacy format")
            # Implementation would depend on the specific format differences
    
    def plot_analysis_results(self, merged, plot_file, bins=10):
        """Create plots using legacy method."""
        return self.legacy_nb.plot_analysis_results(merged, plot_file, bins)


class HybridNetworkBootstrap:
    """
    Hybrid implementation that can use either legacy or OOP implementation.
    
    This class automatically chooses the best available implementation and
    provides a unified interface that works with both.
    """
    
    def __init__(self, config: Optional[Union[dict, 'AnalysisConfig']] = None,
                 prefer_oop: bool = True):
        """
        Initialize hybrid bootstrap.
        
        Args:
            config: Configuration (dict or AnalysisConfig)
            prefer_oop: Whether to prefer OOP implementation when available
        """
        self.prefer_oop = prefer_oop
        self.using_oop = False
        self.using_legacy = False
        
        # Try to initialize OOP implementation
        if OOP_AVAILABLE and prefer_oop:
            try:
                if isinstance(config, dict):
                    oop_config = AnalysisConfig(**config)
                elif config is None:
                    oop_config = AnalysisConfig()
                else:
                    oop_config = config
                
                self.oop_impl = ImprovedNetworkBootstrap(oop_config)
                self.using_oop = True
                self.logger = self.oop_impl.logger
                self.logger.info("Using OOP implementation")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize OOP implementation: {e}")
        
        # Fallback to legacy implementation
        if LEGACY_AVAILABLE:
            try:
                if config and isinstance(config, dict):
                    # Extract legacy parameters from config dict
                    legacy_param = config.get('logger') or config.get('data')
                else:
                    legacy_param = None
                
                legacy_nb = LegacyNetworkBootstrap(legacy_param)
                self.legacy_impl = LegacyNetworkBootstrapAdapter(legacy_nb)
                self.using_legacy = True
                self.logger = logging.getLogger(__name__)
                self.logger.info("Using legacy implementation")
                return
            except Exception as e:
                self.logger.warning(f"Failed to initialize legacy implementation: {e}")
        
        raise RuntimeError("No working implementation available")
    
    def compute_assign_frac(self, df, total_runs=64, inner_group_size=8):
        """Compute assignment fractions using available implementation."""
        if self.using_oop:
            return self.oop_impl.compute_assign_frac(df, total_runs, inner_group_size)
        elif self.using_legacy:
            return self.legacy_impl.compute_assign_frac(df, total_runs, inner_group_size)
        else:
            raise RuntimeError("No implementation available")
    
    def nb_fdr(self, normal_df, shuffled_df, init=64, data_dir=None, fdr=0.05, boot=8):
        """Perform NB-FDR analysis using available implementation."""
        if self.using_oop:
            return self.oop_impl.nb_fdr(normal_df, shuffled_df, init, data_dir, fdr, boot)
        elif self.using_legacy:
            return self.legacy_impl.nb_fdr(normal_df, shuffled_df, init, data_dir, fdr, boot)
        else:
            raise RuntimeError("No implementation available")
    
    def export_results(self, results, txt_file):
        """Export results using available implementation."""
        if self.using_oop:
            return self.oop_impl.export_results(results, txt_file)
        elif self.using_legacy:
            return self.legacy_impl.export_results(results, txt_file)
        else:
            raise RuntimeError("No implementation available")
    
    def plot_analysis_results(self, merged, plot_file, bins=10):
        """Create plots using available implementation."""
        if self.using_oop:
            return self.oop_impl.plot_analysis_results(merged, plot_file, bins)
        elif self.using_legacy:
            return self.legacy_impl.plot_analysis_results(merged, plot_file, bins)
        else:
            raise RuntimeError("No implementation available")
    
    def run_pipeline(self, normal_df, shuffled_df, output_dir=None, **kwargs):
        """Run complete pipeline (OOP only feature)."""
        if self.using_oop:
            return self.oop_impl.run_pipeline(normal_df, shuffled_df, output_dir, **kwargs)
        else:
            self.logger.warning("Pipeline feature only available in OOP implementation")
            # Fallback to basic analysis
            return self.nb_fdr(normal_df, shuffled_df)
    
    def get_implementation_info(self):
        """Get information about which implementation is being used."""
        return {
            'using_oop': self.using_oop,
            'using_legacy': self.using_legacy,
            'oop_available': OOP_AVAILABLE,
            'legacy_available': LEGACY_AVAILABLE,
            'preferred': 'OOP' if self.prefer_oop else 'Legacy'
        }


# Factory function for creating the best available implementation
def create_network_bootstrap(config=None, prefer_oop=True):
    """
    Factory function to create the best available NetworkBootstrap implementation.
    
    Args:
        config: Configuration (dict, AnalysisConfig, or legacy parameter)
        prefer_oop: Whether to prefer OOP implementation
        
    Returns:
        NetworkBootstrap instance (hybrid, OOP, or legacy)
    """
    try:
        return HybridNetworkBootstrap(config, prefer_oop)
    except Exception as e:
        # Final fallback
        if LEGACY_AVAILABLE:
            logging.warning(f"Falling back to basic legacy implementation: {e}")
            return LegacyNetworkBootstrap(config)
        else:
            raise RuntimeError(f"No NetworkBootstrap implementation available: {e}")


# Convenience aliases for backward compatibility
NetworkBootstrap = create_network_bootstrap

def NetworkBootstrapOOP(config=None):
    """Create OOP implementation specifically."""
    if not OOP_AVAILABLE:
        raise ImportError("OOP implementation not available")
    return ImprovedNetworkBootstrap(config)

def NetworkBootstrapLegacy(param=None):
    """Create legacy implementation specifically."""
    if not LEGACY_AVAILABLE:
        raise ImportError("Legacy implementation not available")
    return LegacyNetworkBootstrap(param)
