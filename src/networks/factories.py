"""
Factory pattern implementations for creating network objects.

This module provides concrete factory implementations following the Factory
and Abstract Factory patterns for creating different types of network objects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import requests
import json

from ..core.base import NetworkFactory, AbstractNetwork, AnalysisConfig
from .implementations import NetworkImpl, LegacyNetworkAdapter


class StandardNetworkFactory(NetworkFactory):
    """Standard factory for creating network objects."""
    
    def create_network(self, data: Any, **kwargs) -> AbstractNetwork:
        """Create a network from input data."""
        if isinstance(data, np.ndarray):
            return self._create_from_array(data, **kwargs)
        elif isinstance(data, pd.DataFrame):
            return self._create_from_dataframe(data, **kwargs)
        elif isinstance(data, dict):
            return self._create_from_dict(data, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def create_from_file(self, file_path: Path, **kwargs) -> AbstractNetwork:
        """Create a network from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            return self._create_from_csv(file_path, **kwargs)
        elif file_path.suffix == '.json':
            return self._create_from_json(file_path, **kwargs)
        elif file_path.suffix in ['.npy', '.npz']:
            return self._create_from_numpy(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def create_from_url(self, url: str, **kwargs) -> AbstractNetwork:
        """Create a network from URL."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if url.endswith('.json'):
                data = response.json()
                return self._create_from_dict(data, **kwargs)
            elif url.endswith('.csv'):
                # Save to temp file and read
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = Path(tmp.name)
                return self._create_from_csv(tmp_path, **kwargs)
            else:
                raise ValueError(f"Unsupported URL format: {url}")
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch data from URL: {e}")
    
    def _create_from_array(self, array: np.ndarray, **kwargs) -> AbstractNetwork:
        """Create network from numpy array."""
        node_names = kwargs.get('node_names')
        return NetworkImpl(array, node_names)
    
    def _create_from_dataframe(self, df: pd.DataFrame, **kwargs) -> AbstractNetwork:
        """Create network from pandas DataFrame."""
        # Assume DataFrame is an adjacency matrix representation
        if 'source' in df.columns and 'target' in df.columns:
            return self._create_from_edgelist_df(df, **kwargs)
        else:
            # Assume it's an adjacency matrix
            adj_matrix = df.values
            node_names = df.columns.tolist() if len(df.columns) == len(df) else None
            return NetworkImpl(adj_matrix, node_names)
    
    def _create_from_edgelist_df(self, df: pd.DataFrame, **kwargs) -> AbstractNetwork:
        """Create network from edge list DataFrame."""
        nodes = sorted(set(df['source'].tolist() + df['target'].tolist()))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        n_nodes = len(nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        weight_col = kwargs.get('weight_column', 'weight')
        
        for _, row in df.iterrows():
            i = node_to_idx[row['source']]
            j = node_to_idx[row['target']]
            weight = row.get(weight_col, 1.0)
            adj_matrix[i, j] = weight
        
        return NetworkImpl(adj_matrix, nodes)
    
    def _create_from_dict(self, data: Dict[str, Any], **kwargs) -> AbstractNetwork:
        """Create network from dictionary."""
        if 'adjacency_matrix' in data:
            adj_matrix = np.array(data['adjacency_matrix'])
            node_names = data.get('node_names')
            return NetworkImpl(adj_matrix, node_names)
        elif 'edges' in data:
            return self._create_from_edge_dict(data, **kwargs)
        else:
            raise ValueError("Dictionary must contain 'adjacency_matrix' or 'edges'")
    
    def _create_from_edge_dict(self, data: Dict[str, Any], **kwargs) -> AbstractNetwork:
        """Create network from edge dictionary."""
        edges = data['edges']
        nodes = sorted(set(
            [edge['source'] for edge in edges] + 
            [edge['target'] for edge in edges]
        ))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        n_nodes = len(nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        for edge in edges:
            i = node_to_idx[edge['source']]
            j = node_to_idx[edge['target']]
            weight = edge.get('weight', 1.0)
            adj_matrix[i, j] = weight
        
        return NetworkImpl(adj_matrix, nodes)
    
    def _create_from_csv(self, file_path: Path, **kwargs) -> AbstractNetwork:
        """Create network from CSV file."""
        df = pd.read_csv(file_path)
        return self._create_from_dataframe(df, **kwargs)
    
    def _create_from_json(self, file_path: Path, **kwargs) -> AbstractNetwork:
        """Create network from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self._create_from_dict(data, **kwargs)
    
    def _create_from_numpy(self, file_path: Path, **kwargs) -> AbstractNetwork:
        """Create network from numpy file."""
        if file_path.suffix == '.npy':
            array = np.load(file_path)
        else:  # .npz
            npz_data = np.load(file_path)
            array = npz_data['adjacency_matrix']  # Assume this key exists
        
        return self._create_from_array(array, **kwargs)


class LegacyNetworkFactory(NetworkFactory):
    """Factory for creating networks compatible with legacy code."""
    
    def create_network(self, data: Any, **kwargs) -> AbstractNetwork:
        """Create a legacy-compatible network."""
        # Create standard network first
        standard_factory = StandardNetworkFactory()
        standard_network = standard_factory.create_network(data, **kwargs)
        
        # Wrap in legacy adapter
        return LegacyNetworkAdapter(standard_network.adjacency_matrix)
    
    def create_from_file(self, file_path: Path, **kwargs) -> AbstractNetwork:
        """Create legacy-compatible network from file."""
        standard_factory = StandardNetworkFactory()
        standard_network = standard_factory.create_from_file(file_path, **kwargs)
        return LegacyNetworkAdapter(standard_network.adjacency_matrix)
    
    def create_from_url(self, url: str, **kwargs) -> AbstractNetwork:
        """Create legacy-compatible network from URL."""
        standard_factory = StandardNetworkFactory()
        standard_network = standard_factory.create_from_url(url, **kwargs)
        return LegacyNetworkAdapter(standard_network.adjacency_matrix)


class NetworkFactoryRegistry:
    """Registry for managing different network factories."""
    
    def __init__(self):
        self._factories: Dict[str, NetworkFactory] = {}
        self._default_factory = 'standard'
        
        # Register default factories
        self.register('standard', StandardNetworkFactory())
        self.register('legacy', LegacyNetworkFactory())
    
    def register(self, name: str, factory: NetworkFactory) -> None:
        """Register a new factory."""
        self._factories[name] = factory
    
    def unregister(self, name: str) -> None:
        """Unregister a factory."""
        if name in self._factories:
            del self._factories[name]
    
    def get_factory(self, name: str) -> NetworkFactory:
        """Get a factory by name."""
        if name not in self._factories:
            raise ValueError(f"Factory '{name}' not registered")
        return self._factories[name]
    
    def set_default(self, name: str) -> None:
        """Set the default factory."""
        if name not in self._factories:
            raise ValueError(f"Factory '{name}' not registered")
        self._default_factory = name
    
    def create_network(self, data: Any, factory_name: Optional[str] = None, **kwargs) -> AbstractNetwork:
        """Create network using specified or default factory."""
        factory_name = factory_name or self._default_factory
        factory = self.get_factory(factory_name)
        return factory.create_network(data, **kwargs)
    
    def create_from_file(self, file_path: Path, factory_name: Optional[str] = None, **kwargs) -> AbstractNetwork:
        """Create network from file using specified or default factory."""
        factory_name = factory_name or self._default_factory
        factory = self.get_factory(factory_name)
        return factory.create_from_file(file_path, **kwargs)
    
    def create_from_url(self, url: str, factory_name: Optional[str] = None, **kwargs) -> AbstractNetwork:
        """Create network from URL using specified or default factory."""
        factory_name = factory_name or self._default_factory
        factory = self.get_factory(factory_name)
        return factory.create_from_url(url, **kwargs)
    
    def list_factories(self) -> List[str]:
        """List all registered factories."""
        return list(self._factories.keys())


# Global factory registry instance
network_registry = NetworkFactoryRegistry()


def create_network(data: Any, factory: Optional[str] = None, **kwargs) -> AbstractNetwork:
    """Convenience function to create networks using the global registry."""
    return network_registry.create_network(data, factory, **kwargs)


def create_network_from_file(file_path: Path, factory: Optional[str] = None, **kwargs) -> AbstractNetwork:
    """Convenience function to create networks from file using the global registry."""
    return network_registry.create_from_file(file_path, factory, **kwargs)


def create_network_from_url(url: str, factory: Optional[str] = None, **kwargs) -> AbstractNetwork:
    """Convenience function to create networks from URL using the global registry."""
    return network_registry.create_from_url(url, factory, **kwargs)
