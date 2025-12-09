"""
PyNB Package - Network Bootstrap False Discovery Rate Analysis

This package provides both the original interface and an improved Object-Oriented 
Programming (OOP) architecture for network bootstrap analysis.

Usage:
    # Original interface (backward compatible)
    from pynb import NetworkBootstrap
    nb = NetworkBootstrap()
    
    # New OOP interface  
    from pynb.oop import NetworkBootstrapFacade, AnalysisConfig
    config = AnalysisConfig(total_runs=64, support_threshold=0.8)
    nb = NetworkBootstrapFacade(config)
    
    # Hybrid interface (automatically chooses best implementation)
    from pynb import create_network_bootstrap
    nb = create_network_bootstrap(prefer_oop=True)
"""

# Import legacy interface for backward compatibility
try:
    from .bootstrap.nb_fdr import NetworkBootstrap as LegacyNetworkBootstrap
    from .bootstrap.nb_fdr import NetworkData, NetworkResults
except ImportError:
    LegacyNetworkBootstrap = None
    NetworkData = None  
    NetworkResults = None

# Import new Nestboot class
try:
    from .methods.nestboot import Nestboot
except ImportError:
    Nestboot = None  # type: ignore

# Import integration layer
try:
    from .integration import (
        create_network_bootstrap,
        NetworkBootstrap,
        NetworkBootstrapOOP,
        NetworkBootstrapLegacy,
        HybridNetworkBootstrap
    )
except ImportError:
    # Fallback definitions
    def create_network_bootstrap(*args, **kwargs):
        if LegacyNetworkBootstrap:
            return LegacyNetworkBootstrap(*args, **kwargs)
        else:
            raise ImportError("No NetworkBootstrap implementation available")
    
    NetworkBootstrap = create_network_bootstrap
    NetworkBootstrapOOP = None
    NetworkBootstrapLegacy = LegacyNetworkBootstrap
    HybridNetworkBootstrap = None

# Version information
__version__ = "0.2.0"
__author__ = "Daniel Colin Morgan"
__email__ = "your.email@example.com"

# Export main interface
__all__ = [
    # Main interfaces
    'NetworkBootstrap',
    'create_network_bootstrap',
    
    # Specific implementations
    'NetworkBootstrapOOP', 
    'NetworkBootstrapLegacy',
    'HybridNetworkBootstrap',
    
    # New Nestboot class
    'Nestboot',
    
    # Legacy classes
    'LegacyNetworkBootstrap',
    'NetworkData',
    'NetworkResults',
    
    # Version info
    '__version__',
]

# Package metadata
__doc__ = """
PyNB - Network Bootstrap False Discovery Rate Analysis

A Python package for performing bootstrap-based false discovery rate analysis
on gene regulatory networks. Features both legacy and modern OOP interfaces.

Key Features:
- Bootstrap sampling for network stability assessment
- False Discovery Rate (FDR) control
- Multiple output formats and visualizations  
- Backward compatible API
- Modern OOP architecture with design patterns
- Integration with SCENIC+ workflows

Example Usage:
    import pandas as pd
    from pynb import NetworkBootstrap
    
    # Load your network data
    normal_data = pd.read_csv('normal_network_data.csv')
    shuffled_data = pd.read_csv('shuffled_network_data.csv')
    
    # Create analyzer
    nb = NetworkBootstrap()
    
    # Compute assignment fractions
    normal_agg = nb.compute_assign_frac(normal_data)
    shuffled_agg = nb.compute_assign_frac(shuffled_data)
    
    # Run FDR analysis
    results = nb.nb_fdr(normal_data, shuffled_data, 
                       init=64, fdr=0.05, boot=8)
    
    # Export results
    nb.export_results(results, 'results.txt')

For advanced OOP usage, see the documentation in docs/OOP_ARCHITECTURE.md
"""
