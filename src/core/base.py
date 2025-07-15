"""
Base classes and interfaces for pyNB package.

This module defines abstract base classes and interfaces that establish
the OOP structure for the network bootstrap analysis package.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, TypeVar, Generic, Callable, Union
from dataclasses import dataclass
from pathlib import Path

T = TypeVar('T')

# Type aliases - will be properly typed when numpy/pandas are available
NDArrayFloat = Any
NDArrayBool = Any
DataFrameType = Any

try:
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    NDArrayFloat = npt.NDArray[np.float64]
    NDArrayBool = npt.NDArray[np.bool_]
    DataFrameType = pd.DataFrame
except ImportError:
    # Fallback for when numpy/pandas not available
    np = None
    pd = None


@dataclass
class AnalysisConfig:
    """Configuration for network analysis."""
    total_runs: int = 64
    inner_group_size: int = 8
    support_threshold: float = 0.8
    fdr_threshold: float = 0.05
    epsilon: float = 1e-6


class DataProcessor(Protocol):
    """Protocol for data processing operations."""
    
    def process(self, data: DataFrameType, config: AnalysisConfig) -> DataFrameType:
        """Process input data according to configuration."""
        ...


class NetworkAnalyzer(Protocol):
    """Protocol for network analysis operations."""
    
    def analyze(self, normal_data: DataFrameType, shuffled_data: DataFrameType) -> Any:
        """Analyze network data and return results."""
        ...


class ResultExporter(Protocol):
    """Protocol for exporting analysis results."""
    
    def export(self, results: Any, output_path: Path) -> None:
        """Export results to specified path."""
        ...


class Visualizer(Protocol):
    """Protocol for creating visualizations."""
    
    def visualize(self, data: DataFrameType, output_path: Path, **kwargs) -> None:
        """Create visualization from data."""
        ...


class AbstractNetworkData(ABC):
    """Abstract base class for network data structures."""
    
    def __init__(self, data: Optional[Any] = None):
        self._data = data
        self._metadata: Dict[str, Any] = {}
    
    @property
    def data(self) -> Optional[Any]:
        """Get the underlying data."""
        return self._data
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary."""
        return self._metadata
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the data structure."""
        pass
    
    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        pass


class AbstractNetwork(AbstractNetworkData):
    """Abstract base class for network structures."""
    
    def __init__(self, adjacency_matrix: Optional[Any] = None):
        super().__init__(adjacency_matrix)
        self._node_names: Optional[List[str]] = None
    
    @property
    def adjacency_matrix(self) -> Optional[Any]:
        """Get the adjacency matrix."""
        return self._data
    
    @property
    def node_names(self) -> Optional[List[str]]:
        """Get node names."""
        return self._node_names
    
    @node_names.setter
    def node_names(self, names: List[str]) -> None:
        """Set node names."""
        if self._data is not None and hasattr(self._data, 'shape') and len(names) != self._data.shape[0]:
            raise ValueError("Number of names must match matrix dimensions")
        self._node_names = names
    
    @property
    def num_nodes(self) -> int:
        """Get number of nodes."""
        if self._data is not None and hasattr(self._data, 'shape'):
            return self._data.shape[0]
        return 0
    
    @property
    def num_edges(self) -> int:
        """Get number of edges."""
        if self._data is not None and np is not None:
            return int(np.sum(self._data != 0))
        return 0
    
    @abstractmethod
    def get_density(self) -> float:
        """Calculate network density."""
        pass
    
    @abstractmethod
    def export_format(self, format_type: str) -> Dict[str, Any]:
        """Export network in specified format."""
        pass


class AbstractAnalysisResult(ABC):
    """Abstract base class for analysis results."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._config: Optional[AnalysisConfig] = None
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get computed metrics."""
        return self._metrics
    
    @property
    def config(self) -> Optional[AnalysisConfig]:
        """Get analysis configuration."""
        return self._config
    
    @config.setter
    def config(self, config: AnalysisConfig) -> None:
        """Set analysis configuration."""
        self._config = config
    
    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Get result summary."""
        pass
    
    @abstractmethod
    def validate_results(self) -> bool:
        """Validate result consistency."""
        pass


class AbstractAnalysisStrategy(ABC, Generic[T]):
    """Abstract base class for analysis strategies."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> T:
        """Execute the analysis strategy."""
        pass
    
    @abstractmethod
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input parameters."""
        pass


class NetworkFactory(ABC):
    """Abstract factory for creating network objects."""
    
    @abstractmethod
    def create_network(self, data: Any, **kwargs) -> AbstractNetwork:
        """Create a network from input data."""
        pass
    
    @abstractmethod
    def create_from_file(self, file_path: Path, **kwargs) -> AbstractNetwork:
        """Create a network from file."""
        pass
    
    @abstractmethod
    def create_from_url(self, url: str, **kwargs) -> AbstractNetwork:
        """Create a network from URL."""
        pass


class AnalysisBuilder(ABC):
    """Abstract builder for creating analysis pipelines."""
    
    def __init__(self):
        self.reset()
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the builder state."""
        pass
    
    @abstractmethod
    def set_data_processor(self, processor: DataProcessor) -> 'AnalysisBuilder':
        """Set the data processor."""
        pass
    
    @abstractmethod
    def set_analyzer(self, analyzer: NetworkAnalyzer) -> 'AnalysisBuilder':
        """Set the network analyzer."""
        pass
    
    @abstractmethod
    def set_exporter(self, exporter: ResultExporter) -> 'AnalysisBuilder':
        """Set the result exporter."""
        pass
    
    @abstractmethod
    def set_visualizer(self, visualizer: Visualizer) -> 'AnalysisBuilder':
        """Set the visualizer."""
        pass
    
    @abstractmethod
    def build(self) -> 'AnalysisPipeline':
        """Build the analysis pipeline."""
        pass


class AnalysisPipeline(ABC):
    """Abstract analysis pipeline."""
    
    @abstractmethod
    def run(self, **kwargs) -> AbstractAnalysisResult:
        """Run the complete analysis pipeline."""
        pass
    
    @abstractmethod
    def get_config(self) -> AnalysisConfig:
        """Get pipeline configuration."""
        pass


class Observable:
    """Observable pattern implementation for progress tracking."""
    
    def __init__(self):
        self._observers: List[Callable] = []
    
    def attach(self, observer: Callable) -> None:
        """Attach an observer."""
        self._observers.append(observer)
    
    def detach(self, observer: Callable) -> None:
        """Detach an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify(self, event: str, data: Any = None) -> None:
        """Notify all observers."""
        for observer in self._observers:
            observer(event, data)
