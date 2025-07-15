"""
Concrete implementations of network structures.

This module provides concrete implementations of the abstract network classes
using proper OOP patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import requests
import logging

from ..core.base import AbstractNetwork, AnalysisConfig
from ..datastruct.Exchange import Exchange


class NetworkImpl(AbstractNetwork):
    """Concrete implementation of AbstractNetwork."""
    
    def __init__(self, adjacency_matrix: Optional[np.ndarray] = None, 
                 node_names: Optional[List[str]] = None):
        super().__init__(adjacency_matrix)
        if node_names:
            self.node_names = node_names
        self._network_id = self._generate_id() if adjacency_matrix is not None else ''
    
    def _generate_id(self) -> str:
        """Generate unique network identifier."""
        from datetime import datetime
        N = self.num_nodes
        L = self.num_edges
        ID = str(round(np.random.rand() * 10000))
        return f"dc-D{datetime.now().strftime('%Y%m%d')}-directed-N{N}-L{L}-ID{ID}"
    
    def validate(self) -> bool:
        """Validate the network structure."""
        if self._data is None:
            return False
        
        # Check if it's a square matrix
        if len(self._data.shape) != 2 or self._data.shape[0] != self._data.shape[1]:
            return False
        
        # Check for valid values (no NaN, Inf)
        if np.any(np.isnan(self._data)) or np.any(np.isinf(self._data)):
            return False
        
        return True
    
    def summary(self) -> Dict[str, Any]:
        """Get network summary statistics."""
        if not self.validate():
            return {"error": "Invalid network"}
        
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.get_density(),
            "is_directed": not np.allclose(self._data, self._data.T),
            "has_self_loops": np.any(np.diag(self._data) != 0),
            "network_id": self._network_id
        }
    
    def get_density(self) -> float:
        """Calculate network density."""
        if self._data is None:
            return 0.0
        
        N = self.num_nodes
        if N <= 1:
            return 0.0
        
        return self.num_edges / (N * (N - 1))
    
    def export_format(self, format_type: str) -> Dict[str, Any]:
        """Export network in specified format."""
        if format_type == "adjacency_list":
            return self._to_adjacency_list()
        elif format_type == "edge_list":
            return self._to_edge_list()
        elif format_type == "cytoscape":
            return self._to_cytoscape()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _to_adjacency_list(self) -> Dict[str, Any]:
        """Convert to adjacency list format."""
        adj_list = {}
        for i in range(self.num_nodes):
            neighbors = []
            for j in range(self.num_nodes):
                if self._data[i, j] != 0:
                    node_name = self._node_names[j] if self._node_names else f"node_{j}"
                    neighbors.append({"node": node_name, "weight": float(self._data[i, j])})
            
            source_name = self._node_names[i] if self._node_names else f"node_{i}"
            adj_list[source_name] = neighbors
        
        return {"adjacency_list": adj_list}
    
    def _to_edge_list(self) -> Dict[str, Any]:
        """Convert to edge list format."""
        edges = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self._data[i, j] != 0:
                    source = self._node_names[i] if self._node_names else f"node_{i}"
                    target = self._node_names[j] if self._node_names else f"node_{j}"
                    edges.append({
                        "source": source,
                        "target": target,
                        "weight": float(self._data[i, j])
                    })
        
        return {"edges": edges}
    
    def _to_cytoscape(self) -> Dict[str, Any]:
        """Convert to Cytoscape format."""
        nodes = []
        edges = []
        
        # Add nodes
        for i in range(self.num_nodes):
            node_name = self._node_names[i] if self._node_names else f"node_{i}"
            nodes.append({"data": {"id": node_name, "name": node_name}})
        
        # Add edges
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self._data[i, j] != 0:
                    source = self._node_names[i] if self._node_names else f"node_{i}"
                    target = self._node_names[j] if self._node_names else f"node_{j}"
                    edge_id = f"{source}_{target}"
                    edges.append({
                        "data": {
                            "id": edge_id,
                            "source": source,
                            "target": target,
                            "weight": float(self._data[i, j])
                        }
                    })
        
        return {"elements": {"nodes": nodes, "edges": edges}}


class LegacyNetworkAdapter(NetworkImpl):
    """Adapter to make legacy Network class compatible with new OOP structure."""
    
    def __init__(self, legacy_network: Optional['Network'] = None):
        if legacy_network is not None:
            super().__init__(legacy_network.A, None)
            self._legacy_network = legacy_network
        else:
            super().__init__()
            self._legacy_network = None
    
    @classmethod
    def from_legacy(cls, legacy_network: 'Network') -> 'LegacyNetworkAdapter':
        """Create adapter from legacy Network object."""
        return cls(legacy_network)
    
    def to_legacy(self) -> 'Network':
        """Convert back to legacy Network object."""
        from ..datastruct.Network import Network
        network = Network(self._data)
        return network


class NetworkComposite(AbstractNetwork):
    """Composite pattern implementation for handling multiple networks."""
    
    def __init__(self):
        super().__init__(None)
        self._networks: List[AbstractNetwork] = []
        self._weights: List[float] = []
    
    def add_network(self, network: AbstractNetwork, weight: float = 1.0) -> None:
        """Add a network to the composite."""
        self._networks.append(network)
        self._weights.append(weight)
    
    def remove_network(self, network: AbstractNetwork) -> None:
        """Remove a network from the composite."""
        if network in self._networks:
            index = self._networks.index(network)
            self._networks.pop(index)
            self._weights.pop(index)
    
    def get_network_count(self) -> int:
        """Get number of networks in composite."""
        return len(self._networks)
    
    def validate(self) -> bool:
        """Validate all networks in composite."""
        return all(network.validate() for network in self._networks)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all networks."""
        summaries = [network.summary() for network in self._networks]
        return {
            "network_count": self.get_network_count(),
            "individual_summaries": summaries,
            "total_nodes": sum(s.get("num_nodes", 0) for s in summaries),
            "total_edges": sum(s.get("num_edges", 0) for s in summaries)
        }
    
    def get_density(self) -> float:
        """Calculate average density across all networks."""
        if not self._networks:
            return 0.0
        
        densities = [network.get_density() for network in self._networks]
        return np.average(densities, weights=self._weights)
    
    def export_format(self, format_type: str) -> Dict[str, Any]:
        """Export all networks in specified format."""
        exports = []
        for i, network in enumerate(self._networks):
            export_data = network.export_format(format_type)
            export_data["network_index"] = i
            export_data["weight"] = self._weights[i]
            exports.append(export_data)
        
        return {"composite_networks": exports}
    
    @property
    def num_nodes(self) -> int:
        """Get total number of unique nodes across all networks."""
        all_nodes = set()
        for network in self._networks:
            if network.node_names:
                all_nodes.update(network.node_names)
            else:
                all_nodes.update(f"node_{i}" for i in range(network.num_nodes))
        return len(all_nodes)
    
    @property
    def num_edges(self) -> int:
        """Get total number of edges across all networks."""
        return sum(network.num_edges for network in self._networks)


class NetworkProxy(AbstractNetwork):
    """Proxy pattern for lazy loading of large networks."""
    
    def __init__(self, data_source: str, loader_func: callable):
        super().__init__(None)
        self._data_source = data_source
        self._loader_func = loader_func
        self._loaded = False
        self._real_network: Optional[AbstractNetwork] = None
    
    def _ensure_loaded(self) -> None:
        """Ensure the network data is loaded."""
        if not self._loaded:
            self._real_network = self._loader_func(self._data_source)
            self._loaded = True
    
    def validate(self) -> bool:
        """Validate the network (loads if necessary)."""
        self._ensure_loaded()
        return self._real_network.validate() if self._real_network else False
    
    def summary(self) -> Dict[str, Any]:
        """Get network summary (loads if necessary)."""
        self._ensure_loaded()
        summary = self._real_network.summary() if self._real_network else {}
        summary["data_source"] = self._data_source
        summary["loaded"] = self._loaded
        return summary
    
    def get_density(self) -> float:
        """Calculate density (loads if necessary)."""
        self._ensure_loaded()
        return self._real_network.get_density() if self._real_network else 0.0
    
    def export_format(self, format_type: str) -> Dict[str, Any]:
        """Export format (loads if necessary)."""
        self._ensure_loaded()
        return self._real_network.export_format(format_type) if self._real_network else {}
    
    @property
    def adjacency_matrix(self) -> Optional[Any]:
        """Get adjacency matrix (loads if necessary)."""
        self._ensure_loaded()
        return self._real_network.adjacency_matrix if self._real_network else None
    
    @property
    def num_nodes(self) -> int:
        """Get number of nodes (loads if necessary)."""
        self._ensure_loaded()
        return self._real_network.num_nodes if self._real_network else 0
    
    @property
    def num_edges(self) -> int:
        """Get number of edges (loads if necessary)."""
        self._ensure_loaded()
        return self._real_network.num_edges if self._real_network else 0
