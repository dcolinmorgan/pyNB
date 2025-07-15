"""
OOP namespace for pyNB package.

This module provides access to the Object-Oriented Programming components
of the pyNB package, including design patterns and modern architecture.

Usage:
    from pynb.oop import AnalysisConfig, NetworkBootstrapFacade
    from pynb.oop import create_standard_pipeline, create_network
    
    config = AnalysisConfig(total_runs=64, support_threshold=0.8)
    nb = NetworkBootstrapFacade(config)
    results = nb.quick_analysis('normal.csv', 'shuffled.csv', 'output/')
"""

# Core OOP components
try:
    from .core.base import (
        AnalysisConfig,
        AbstractNetwork,
        AbstractAnalysisResult,
        AbstractAnalysisStrategy,
        NetworkFactory,
        AnalysisBuilder,
        AnalysisPipeline,
        Observable,
        DataProcessor,
        NetworkAnalyzer,
        ResultExporter,
        Visualizer,
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    AnalysisConfig = None

# Network components
try:
    from .networks import (
        NetworkImpl,
        LegacyNetworkAdapter,
        NetworkComposite,
        NetworkProxy,
        StandardNetworkFactory,
        LegacyNetworkFactory,
        NetworkFactoryRegistry,
        create_network,
        create_network_from_file,
        create_network_from_url,
    )
    NETWORKS_AVAILABLE = True
except ImportError:
    NETWORKS_AVAILABLE = False
    create_network = None

# Analysis components  
try:
    from .analysis import (
        NBFDRAnalysisResult,
        NBFDRStrategy,
        BootstrapSamplingStrategy,
        NetworkComparisonStrategy,
        AnalysisStrategyFactory,
        StandardDataProcessor,
        NBFDRAnalyzer,
        TextResultExporter,
        MatplotlibVisualizer,
        NBFDRPipeline,
        NBFDRPipelineBuilder,
        PipelineDirector,
        create_standard_pipeline,
        create_minimal_pipeline,
    )
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    create_standard_pipeline = None

# Main OOP interface
try:
    from .improved_bootstrap import (
        ImprovedNetworkBootstrap,
        NetworkBootstrapFacade,
    )
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False
    NetworkBootstrapFacade = None

# Integration components
try:
    from .integration import (
        HybridNetworkBootstrap,
        LegacyResultsAdapter,
        LegacyNetworkBootstrapAdapter,
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

def get_availability_status():
    """Get the availability status of OOP components."""
    return {
        'core': CORE_AVAILABLE,
        'networks': NETWORKS_AVAILABLE,
        'analysis': ANALYSIS_AVAILABLE,
        'bootstrap': BOOTSTRAP_AVAILABLE,
        'integration': INTEGRATION_AVAILABLE,
    }

def check_oop_setup():
    """Check if OOP components are properly set up."""
    status = get_availability_status()
    all_available = all(status.values())
    
    if all_available:
        print("✅ All OOP components are available")
    else:
        print("⚠️  Some OOP components are not available:")
        for component, available in status.items():
            icon = "✅" if available else "❌"
            print(f"   {icon} {component}")
    
    return all_available

# Export main OOP interface
__all__ = [
    # Main interfaces
    'AnalysisConfig',
    'NetworkBootstrapFacade',
    'ImprovedNetworkBootstrap',
    
    # Core abstractions
    'AbstractNetwork',
    'AbstractAnalysisResult', 
    'AbstractAnalysisStrategy',
    'NetworkFactory',
    'AnalysisBuilder',
    'AnalysisPipeline',
    'Observable',
    
    # Protocols
    'DataProcessor',
    'NetworkAnalyzer',
    'ResultExporter',
    'Visualizer',
    
    # Network implementations
    'NetworkImpl',
    'LegacyNetworkAdapter',
    'NetworkComposite',
    'NetworkProxy',
    
    # Factories
    'StandardNetworkFactory',
    'LegacyNetworkFactory',
    'NetworkFactoryRegistry',
    'create_network',
    'create_network_from_file',
    'create_network_from_url',
    
    # Analysis strategies
    'NBFDRAnalysisResult',
    'NBFDRStrategy',
    'BootstrapSamplingStrategy',
    'NetworkComparisonStrategy',
    'AnalysisStrategyFactory',
    
    # Pipeline components
    'StandardDataProcessor',
    'NBFDRAnalyzer',
    'TextResultExporter',
    'MatplotlibVisualizer',
    'NBFDRPipeline',
    'NBFDRPipelineBuilder',
    'PipelineDirector',
    
    # Pipeline factories
    'create_standard_pipeline',
    'create_minimal_pipeline',
    
    # Integration
    'HybridNetworkBootstrap',
    'LegacyResultsAdapter',
    'LegacyNetworkBootstrapAdapter',
    
    # Utilities
    'get_availability_status',
    'check_oop_setup',
]

# Package info
__doc__ = """
PyNB OOP Architecture

This module provides access to the Object-Oriented Programming components
of the pyNB package. The OOP architecture includes:

1. **Design Patterns**:
   - Factory Pattern: Flexible object creation
   - Strategy Pattern: Interchangeable algorithms  
   - Builder Pattern: Complex object construction
   - Facade Pattern: Simplified interface
   - Observer Pattern: Progress tracking
   - Adapter Pattern: Legacy compatibility
   - Composite Pattern: Hierarchical structures

2. **Core Components**:
   - AnalysisConfig: Centralized configuration
   - Abstract base classes: Clear interfaces
   - Protocol definitions: Typing contracts

3. **Network Management**:
   - Multiple network implementations
   - Factory-based creation
   - Format conversion utilities

4. **Analysis Pipeline**:
   - Strategy-based analysis methods
   - Builder-based pipeline construction
   - Progress tracking and monitoring

5. **Integration**:
   - Backward compatibility adapters
   - Hybrid implementations
   - Legacy code integration

Example Usage:
    # Simple facade usage
    from pynb.oop import AnalysisConfig, NetworkBootstrapFacade
    
    config = AnalysisConfig(total_runs=64, support_threshold=0.8)
    nb = NetworkBootstrapFacade(config)
    results = nb.quick_analysis('normal.csv', 'shuffled.csv', 'output/')
    
    # Advanced pipeline usage
    from pynb.oop import create_standard_pipeline
    
    pipeline = create_standard_pipeline(config)
    pipeline.attach(lambda event, data: print(f"Progress: {event}"))
    results = pipeline.run(normal_df, shuffled_df, output_dir)
    
    # Network creation with factory
    from pynb.oop import create_network
    
    network = create_network(adjacency_matrix, node_names=genes)
    edge_list = network.export_format('edge_list')

For complete documentation, see docs/OOP_ARCHITECTURE.md
"""
