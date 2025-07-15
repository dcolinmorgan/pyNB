"""
Builder pattern implementations for creating complex analysis pipelines.

This module provides concrete builder implementations following the Builder
pattern for constructing complex analysis workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from ..core.base import (
    AnalysisBuilder, AnalysisPipeline, AbstractAnalysisResult, AnalysisConfig,
    DataProcessor, NetworkAnalyzer, ResultExporter, Visualizer, Observable
)
from .strategies import NBFDRStrategy, BootstrapSamplingStrategy, AnalysisStrategyFactory


class StandardDataProcessor:
    """Standard implementation of data processor."""
    
    def process(self, data: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
        """Process input data according to configuration."""
        # Basic data cleaning and validation
        processed = data.copy()
        
        # Remove duplicates
        processed = processed.drop_duplicates()
        
        # Filter based on configuration
        if 'run' in processed.columns:
            processed = processed[
                processed['run'].astype(str).str.extract(r'(\d+)')[0].astype(int) < config.total_runs
            ]
        
        # Sort for consistency
        if all(col in processed.columns for col in ['gene_i', 'gene_j', 'run']):
            processed = processed.sort_values(['gene_i', 'gene_j', 'run'])
        
        return processed


class NBFDRAnalyzer:
    """Network analyzer using NB-FDR strategy."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.strategy = NBFDRStrategy(config)
    
    def analyze(self, normal_data: pd.DataFrame, shuffled_data: pd.DataFrame) -> Any:
        """Analyze network data and return results."""
        return self.strategy.execute(normal_data, shuffled_data)


class TextResultExporter:
    """Exporter for creating text-based result summaries."""
    
    def export(self, results: Any, output_path: Path) -> None:
        """Export results to text file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("Network Bootstrap FDR Analysis Results\n")
            f.write("=" * 40 + "\n")
            
            if hasattr(results, 'get_summary'):
                summary = results.get_summary()
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
            
            if hasattr(results, 'metrics'):
                metrics = results.metrics
                f.write(f"\nDetailed Metrics:\n")
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        f.write(f"{key} shape: {value.shape}\n")
                    else:
                        f.write(f"{key}: {value}\n")


class MatplotlibVisualizer:
    """Visualizer using matplotlib for creating plots."""
    
    def visualize(self, data: pd.DataFrame, output_path: Path, **kwargs) -> None:
        """Create visualization from data."""
        try:
            import matplotlib.pyplot as plt
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simple comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot assignment fractions if available
            if 'Afrac_norm' in data.columns and 'Afrac_shuf' in data.columns:
                ax1.hist(data['Afrac_norm'], bins=kwargs.get('bins', 20), 
                        alpha=0.7, label='Normal', color='blue')
                ax1.hist(data['Afrac_shuf'], bins=kwargs.get('bins', 20), 
                        alpha=0.7, label='Shuffled', color='red')
                ax1.set_xlabel('Assignment Fraction')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Assignment Fraction Distribution')
                ax1.legend()
            
            # Plot sign fractions if available
            if 'Asign_frac_norm' in data.columns and 'Asign_frac_shuf' in data.columns:
                ax2.hist(data['Asign_frac_norm'], bins=kwargs.get('bins', 20), 
                        alpha=0.7, label='Normal', color='blue')
                ax2.hist(data['Asign_frac_shuf'], bins=kwargs.get('bins', 20), 
                        alpha=0.7, label='Shuffled', color='red')
                ax2.set_xlabel('Sign Fraction')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Sign Fraction Distribution')
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            # Fallback if matplotlib not available
            print(f"Matplotlib not available. Skipping visualization to {output_path}")


class NBFDRPipeline(AnalysisPipeline, Observable):
    """Concrete analysis pipeline for NB-FDR analysis."""
    
    def __init__(self, config: AnalysisConfig,
                 data_processor: Optional[DataProcessor] = None,
                 analyzer: Optional[NetworkAnalyzer] = None,
                 exporter: Optional[ResultExporter] = None,
                 visualizer: Optional[Visualizer] = None):
        Observable.__init__(self)
        self.config = config
        self.data_processor = data_processor or StandardDataProcessor()
        self.analyzer = analyzer or NBFDRAnalyzer(config)
        self.exporter = exporter or TextResultExporter()
        self.visualizer = visualizer or MatplotlibVisualizer()
        self.logger = logging.getLogger(__name__)
    
    def run(self, normal_data: pd.DataFrame, shuffled_data: pd.DataFrame,
            output_dir: Optional[Path] = None, **kwargs) -> AbstractAnalysisResult:
        """Run the complete analysis pipeline."""
        try:
            self.notify("pipeline_started", {"stage": "initialization"})
            
            # Step 1: Process data
            self.logger.info("Processing input data")
            self.notify("stage_started", {"stage": "data_processing"})
            
            processed_normal = self.data_processor.process(normal_data, self.config)
            processed_shuffled = self.data_processor.process(shuffled_data, self.config)
            
            self.notify("stage_completed", {"stage": "data_processing"})
            
            # Step 2: Analyze
            self.logger.info("Running analysis")
            self.notify("stage_started", {"stage": "analysis"})
            
            results = self.analyzer.analyze(processed_normal, processed_shuffled)
            
            self.notify("stage_completed", {"stage": "analysis"})
            
            # Step 3: Export results (if output directory provided)
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                self.logger.info("Exporting results")
                self.notify("stage_started", {"stage": "export"})
                
                result_file = output_dir / "results.txt"
                self.exporter.export(results, result_file)
                
                self.notify("stage_completed", {"stage": "export"})
                
                # Step 4: Create visualizations
                self.logger.info("Creating visualizations")
                self.notify("stage_started", {"stage": "visualization"})
                
                # Prepare data for visualization
                normal_agg = self._compute_assign_frac(processed_normal)
                shuffled_agg = self._compute_assign_frac(processed_shuffled)
                
                normal_agg.rename(columns={
                    'Afrac': 'Afrac_norm',
                    'Asign_frac': 'Asign_frac_norm'
                }, inplace=True)
                shuffled_agg.rename(columns={
                    'Afrac': 'Afrac_shuf',
                    'Asign_frac': 'Asign_frac_shuf'
                }, inplace=True)
                
                merged = pd.merge(normal_agg, shuffled_agg, on=['gene_i', 'gene_j'])
                
                plot_file = output_dir / "analysis_plot.png"
                self.visualizer.visualize(merged, plot_file, **kwargs)
                
                self.notify("stage_completed", {"stage": "visualization"})
            
            self.notify("pipeline_completed", {"results": results})
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.notify("pipeline_failed", {"error": str(e)})
            raise
    
    def get_config(self) -> AnalysisConfig:
        """Get pipeline configuration."""
        return self.config
    
    def _compute_assign_frac(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute assignment fractions for visualization."""
        # Simplified version for visualization
        strategy = NBFDRStrategy(self.config)
        return strategy._compute_assign_frac(df)


class NBFDRPipelineBuilder(AnalysisBuilder):
    """Builder for creating NB-FDR analysis pipelines."""
    
    def __init__(self):
        super().__init__()
        self.reset()
    
    def reset(self) -> None:
        """Reset the builder state."""
        self._config = AnalysisConfig()
        self._data_processor: Optional[DataProcessor] = None
        self._analyzer: Optional[NetworkAnalyzer] = None
        self._exporter: Optional[ResultExporter] = None
        self._visualizer: Optional[Visualizer] = None
    
    def set_config(self, config: AnalysisConfig) -> 'NBFDRPipelineBuilder':
        """Set the analysis configuration."""
        self._config = config
        return self
    
    def set_data_processor(self, processor: DataProcessor) -> 'NBFDRPipelineBuilder':
        """Set the data processor."""
        self._data_processor = processor
        return self
    
    def set_analyzer(self, analyzer: NetworkAnalyzer) -> 'NBFDRPipelineBuilder':
        """Set the network analyzer."""
        self._analyzer = analyzer
        return self
    
    def set_exporter(self, exporter: ResultExporter) -> 'NBFDRPipelineBuilder':
        """Set the result exporter."""
        self._exporter = exporter
        return self
    
    def set_visualizer(self, visualizer: Visualizer) -> 'NBFDRPipelineBuilder':
        """Set the visualizer."""
        self._visualizer = visualizer
        return self
    
    def build(self) -> NBFDRPipeline:
        """Build the analysis pipeline."""
        pipeline = NBFDRPipeline(
            config=self._config,
            data_processor=self._data_processor,
            analyzer=self._analyzer,
            exporter=self._exporter,
            visualizer=self._visualizer
        )
        
        # Reset for next build
        self.reset()
        return pipeline


class PipelineDirector:
    """Director for orchestrating pipeline construction."""
    
    def __init__(self, builder: AnalysisBuilder):
        self.builder = builder
    
    def create_standard_pipeline(self, config: Optional[AnalysisConfig] = None) -> AnalysisPipeline:
        """Create a standard NB-FDR pipeline."""
        config = config or AnalysisConfig()
        
        return (self.builder
                .set_config(config)
                .set_data_processor(StandardDataProcessor())
                .set_analyzer(NBFDRAnalyzer(config))
                .set_exporter(TextResultExporter())
                .set_visualizer(MatplotlibVisualizer())
                .build())
    
    def create_minimal_pipeline(self, config: Optional[AnalysisConfig] = None) -> AnalysisPipeline:
        """Create a minimal pipeline with only analysis."""
        config = config or AnalysisConfig()
        
        return (self.builder
                .set_config(config)
                .set_analyzer(NBFDRAnalyzer(config))
                .build())
    
    def create_custom_pipeline(self, **components) -> AnalysisPipeline:
        """Create a custom pipeline with specified components."""
        builder = self.builder.reset()
        
        if 'config' in components:
            builder = builder.set_config(components['config'])
        if 'data_processor' in components:
            builder = builder.set_data_processor(components['data_processor'])
        if 'analyzer' in components:
            builder = builder.set_analyzer(components['analyzer'])
        if 'exporter' in components:
            builder = builder.set_exporter(components['exporter'])
        if 'visualizer' in components:
            builder = builder.set_visualizer(components['visualizer'])
        
        return builder.build()


# Convenience functions for common use cases
def create_standard_pipeline(config: Optional[AnalysisConfig] = None) -> NBFDRPipeline:
    """Create a standard NB-FDR analysis pipeline."""
    builder = NBFDRPipelineBuilder()
    director = PipelineDirector(builder)
    return director.create_standard_pipeline(config)


def create_minimal_pipeline(config: Optional[AnalysisConfig] = None) -> NBFDRPipeline:
    """Create a minimal analysis pipeline."""
    builder = NBFDRPipelineBuilder()
    director = PipelineDirector(builder)
    return director.create_minimal_pipeline(config)
