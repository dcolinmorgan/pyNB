"""
Example demonstrating improved OOP usage of the pyNB package.

This example shows how to use the new OOP architecture with design patterns
while maintaining backward compatibility with the existing API.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import the improved OOP components
try:
    from src.core.base import AnalysisConfig
    from src.improved_bootstrap import ImprovedNetworkBootstrap, NetworkBootstrapFacade
    from src.networks.factories import create_network, network_registry
    from src.analysis.strategies import AnalysisStrategyFactory
    from src.analysis.builders import create_standard_pipeline, NBFDRPipelineBuilder, PipelineDirector
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This example requires the improved OOP modules to be properly set up.")
    exit(1)


def demonstrate_improved_oop_usage():
    """Demonstrate the improved OOP usage patterns."""
    
    print("=" * 60)
    print("PyNB Improved OOP Architecture Demonstration")
    print("=" * 60)
    
    # 1. Configuration Management
    print("\n1. Configuration Management with AnalysisConfig")
    print("-" * 50)
    
    # Create configuration object
    config = AnalysisConfig(
        total_runs=64,
        inner_group_size=8,
        support_threshold=0.8,
        fdr_threshold=0.05,
        epsilon=1e-6
    )
    print(f"Created config: total_runs={config.total_runs}, support_threshold={config.support_threshold}")
    
    
    # 2. Factory Pattern for Network Creation
    print("\n2. Factory Pattern for Network Creation")
    print("-" * 50)
    
    # Create a sample adjacency matrix
    np.random.seed(42)
    adj_matrix = np.random.rand(5, 5)
    adj_matrix[adj_matrix < 0.7] = 0  # Make sparse
    node_names = [f"Gene_{i}" for i in range(5)]
    
    # Use factory to create networks
    try:
        network = create_network(adj_matrix, node_names=node_names)
        print(f"Created network with {network.num_nodes} nodes and {network.num_edges} edges")
        print(f"Network density: {network.get_density():.3f}")
        
        # Export in different formats
        edge_list = network.export_format("edge_list")
        print(f"Exported {len(edge_list['edges'])} edges in edge list format")
    except Exception as e:
        print(f"Network creation demo failed: {e}")
    
    
    # 3. Strategy Pattern for Different Analysis Methods
    print("\n3. Strategy Pattern for Analysis Methods")
    print("-" * 50)
    
    # Create sample data
    sample_data = create_sample_network_data()
    
    # Use strategy factory
    try:
        strategy_factory = AnalysisStrategyFactory()
        
        # Create NB-FDR strategy
        nb_fdr_strategy = strategy_factory.create_strategy('nb_fdr', config)
        print(f"Created strategy: {type(nb_fdr_strategy).__name__}")
        
        # Create bootstrap sampling strategy
        bootstrap_strategy = strategy_factory.create_strategy('bootstrap_sampling', config)
        print(f"Created strategy: {type(bootstrap_strategy).__name__}")
        
        # List available strategies
        available_strategies = strategy_factory.list_strategies()
        print(f"Available strategies: {available_strategies}")
    except Exception as e:
        print(f"Strategy pattern demo failed: {e}")
    
    
    # 4. Builder Pattern for Pipeline Construction
    print("\n4. Builder Pattern for Pipeline Construction")
    print("-" * 50)
    
    try:
        # Method 1: Use convenience function
        pipeline1 = create_standard_pipeline(config)
        print(f"Created standard pipeline: {type(pipeline1).__name__}")
        
        # Method 2: Use builder directly
        builder = NBFDRPipelineBuilder()
        pipeline2 = (builder
                     .set_config(config)
                     .set_data_processor(StandardDataProcessor())
                     .set_analyzer(NBFDRAnalyzer(config))
                     .build())
        print(f"Created custom pipeline: {type(pipeline2).__name__}")
        
        # Method 3: Use director
        director = PipelineDirector(NBFDRPipelineBuilder())
        pipeline3 = director.create_minimal_pipeline(config)
        print(f"Created minimal pipeline: {type(pipeline3).__name__}")
    except Exception as e:
        print(f"Builder pattern demo failed: {e}")
    
    
    # 5. Facade Pattern for Simplified Interface
    print("\n5. Facade Pattern for Simplified Interface")
    print("-" * 50)
    
    try:
        # Create facade with configuration
        facade = NetworkBootstrapFacade(config)
        print(f"Created facade with config: {facade.config.total_runs} runs")
        
        # Use legacy-style configuration
        legacy_facade = NetworkBootstrapFacade.from_legacy_config(
            total_runs=32, support_threshold=0.9
        )
        print(f"Created legacy facade: {legacy_facade.config.total_runs} runs, {legacy_facade.config.support_threshold} threshold")
        
        # Demonstrate backward compatibility
        improved_nb = ImprovedNetworkBootstrap(config)
        
        # Use improved interface
        normal_data, shuffled_data = create_sample_dataframes()
        assign_frac = improved_nb.compute_assign_frac(normal_data)
        print(f"Computed assignment fractions: {len(assign_frac)} gene pairs")
    except Exception as e:
        print(f"Facade pattern demo failed: {e}")
    
    
    # 6. Observer Pattern for Progress Tracking
    print("\n6. Observer Pattern for Progress Tracking")
    print("-" * 50)
    
    try:
        # Create pipeline with observer
        pipeline = create_standard_pipeline(config)
        
        # Add progress observer
        def progress_observer(event, data):
            print(f"  Progress: {event} - {data}")
        
        pipeline.attach(progress_observer)
        print("Added progress observer to pipeline")
        
        # Simulate some events
        pipeline.notify("pipeline_started", {"stage": "initialization"})
        pipeline.notify("stage_completed", {"stage": "data_processing"})
        print("Demonstrated observer notifications")
    except Exception as e:
        print(f"Observer pattern demo failed: {e}")
    
    
    # 7. Full Analysis Workflow
    print("\n7. Complete Analysis Workflow")
    print("-" * 50)
    
    try:
        # Create test data
        normal_df, shuffled_df = create_sample_dataframes()
        
        # Run complete analysis using facade
        facade = NetworkBootstrapFacade(config)
        
        # Option 1: Simple analysis
        results = facade.nb_fdr(normal_df, shuffled_df)
        print(f"Analysis completed. Results type: {type(results).__name__}")
        
        if hasattr(results, 'get_summary'):
            summary = results.get_summary()
            print(f"Result summary: {summary}")
        
        # Option 2: Full pipeline with output
        output_dir = Path("demo_output")
        output_dir.mkdir(exist_ok=True)
        
        pipeline_results = facade.run_pipeline(
            normal_df, shuffled_df, output_dir, pipeline_type='standard'
        )
        print(f"Pipeline completed with output to {output_dir}")
        
    except Exception as e:
        print(f"Full workflow demo failed: {e}")
    
    
    print("\n" + "=" * 60)
    print("OOP Demonstration Complete!")
    print("=" * 60)


def create_sample_network_data():
    """Create sample network data for demonstration."""
    np.random.seed(42)
    
    # Create adjacency matrices
    n_nodes = 10
    matrices = []
    
    for i in range(5):
        matrix = np.random.rand(n_nodes, n_nodes)
        matrix[matrix < 0.8] = 0  # Make sparse
        matrices.append(matrix)
    
    return matrices


def create_sample_dataframes():
    """Create sample DataFrames for analysis."""
    np.random.seed(42)
    
    # Generate sample gene interaction data
    genes = [f"Gene_{i}" for i in range(20)]
    
    data = []
    for run in range(64):
        for i, gene_i in enumerate(genes):
            for j, gene_j in enumerate(genes):
                if i != j and np.random.rand() > 0.95:  # Sparse interactions
                    link_value = np.random.randn()
                    data.append({
                        'gene_i': gene_i,
                        'gene_j': gene_j,
                        'run': f"run_{run}",
                        'link_value': link_value
                    })
    
    df = pd.DataFrame(data)
    
    # Create normal and shuffled versions
    normal_df = df.copy()
    shuffled_df = df.copy()
    
    # Shuffle the link values for shuffled data
    shuffled_values = shuffled_df['link_value'].values.copy()
    np.random.shuffle(shuffled_values)
    shuffled_df['link_value'] = shuffled_values
    
    return normal_df, shuffled_df


def demonstrate_legacy_compatibility():
    """Demonstrate backward compatibility with existing code."""
    
    print("\n" + "=" * 60)
    print("Legacy Compatibility Demonstration")
    print("=" * 60)
    
    try:
        from src.improved_bootstrap import NetworkBootstrap
        
        # Create using legacy interface
        nb = NetworkBootstrap()
        print("Created NetworkBootstrap using legacy interface")
        
        # Use legacy-style methods
        normal_df, shuffled_df = create_sample_dataframes()
        
        # Legacy method calls work the same way
        assign_frac = nb.compute_assign_frac(normal_df, total_runs=32)
        print(f"Legacy method call successful: {len(assign_frac)} results")
        
        # But now with improved capabilities
        results = nb.run_pipeline(normal_df, shuffled_df, pipeline_type='minimal')
        print(f"Enhanced capabilities available: {type(results).__name__}")
        
    except Exception as e:
        print(f"Legacy compatibility demo failed: {e}")


if __name__ == "__main__":
    try:
        demonstrate_improved_oop_usage()
        demonstrate_legacy_compatibility()
    except Exception as e:
        print(f"Demonstration failed: {e}")
        print("\nNote: This demonstration requires all the OOP modules to be properly implemented.")
        print("Some import errors are expected in the current development state.")
