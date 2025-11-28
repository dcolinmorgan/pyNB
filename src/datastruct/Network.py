import numpy as np
import os
from datetime import datetime
from math import floor
import requests
import json
from typing import Optional, Dict, Any, List, Union
from datastruct.Exchange import Exchange

class Network(Exchange):
    """Stores a network matrix A and calculates important network properties."""
    
    def __init__(self, A: np.ndarray = None, network_type: str = 'unknown'):
        super().__init__()
        self._A = None
        self._G = None  # Static gain model
        self._network = ''
        self._names: List[str] = []
        self.description = ''
        self.tol = np.finfo(float).eps
        self.created = {
            'creator': os.getenv('USER') or os.getenv('USERNAME') or '',
            'time': datetime.now(),
            'id': '',
            'nodes': '',
            'type': network_type,
            'sparsity': ''
        }
        
        if A is not None:
            self.setA(A)
            self.setname()
    
    def setA(self, A: np.ndarray):
        """Set the adjacency matrix and compute derived properties."""
        self._A = A
        self._G = -np.linalg.pinv(A.astype(float))
        
        cond_val = np.linalg.cond(A.astype(float))
        if np.isinf(cond_val) or np.isnan(cond_val):
            id_val = 'inf'
        else:
            id_val = str(round(cond_val * 10000))
        self.created['id'] = id_val
        self.created['nodes'] = str(A.shape[0])
        self.created['sparsity'] = str(np.count_nonzero(A))
    
    def setname(self, namestruct: Optional[Dict] = None):
        """Set the network name based on created properties."""
        if namestruct is None:
            namestruct = self.created
        elif not isinstance(namestruct, dict):
            raise ValueError('Input must be a dict')
        
        # Update created with namestruct
        for key, value in namestruct.items():
            if key in self.created:
                self.created[key] = value
        
        namer = self.created
        self._network = f"{namer['creator']}-D{datetime.now().strftime('%Y%m%d')}-{namer['type']}-N{namer['nodes']}-L{np.count_nonzero(self._A)}-ID{namer['id']}"
    
    @property
    def A(self) -> np.ndarray:
        return self._A
    
    @property
    def G(self) -> np.ndarray:
        return self._G
    
    @property
    def network(self) -> str:
        return self._network
    
    @network.setter
    def network(self, value: str):
        self._network = value
    
    @property
    def N(self) -> int:
        """Number of nodes."""
        return self._A.shape[0] if self._A is not None else 0
    
    @property
    def names(self) -> List[str]:
        """Node names, generates defaults if empty."""
        if not self._names:
            for i in range(1, self.N + 1):
                self._names.append(f"G{i:0{floor(np.log10(self.N)) + 1}d}")
        return self._names
    
    @names.setter
    def names(self, value: List[str]):
        self._names = value
    
    def show(self):
        """Display network matrix and properties (text-based approximation)."""
        if self._A is None:
            print("No network matrix to display")
            return
        
        print("Network Matrix:")
        print(self._A)
        print("\nNetwork Properties:")
        print(f"Name: {self.network}")
        print(f"Description: {self.description}")
        print(f"Sparseness: {np.count_nonzero(self._A) / self._A.size}")
        print(f"# Nodes: {self._A.shape[0]}")
        print(f"# Links: {np.count_nonzero(self._A)}")
    
    def view(self):
        """Graphical network view (placeholder - would need networkx/matplotlib)."""
        print("Network visualization not implemented in this Python version")
        print("Consider using networkx for graph visualization")
    
    def sign(self) -> np.ndarray:
        """Return sign of adjacency matrix."""
        if self._A is None:
            raise ValueError("Network matrix not set")
        return np.sign(self._A)
    
    def logical(self) -> np.ndarray:
        """Return logical (boolean) version of adjacency matrix."""
        if self._A is None:
            raise ValueError("Network matrix not set")
        return self._A.astype(bool)
    
    def size(self, dim: Optional[int] = None) -> Union[tuple, int]:
        """Return size of adjacency matrix."""
        if self._A is None:
            raise ValueError("Network matrix not set")
        if dim is not None:
            return self._A.shape[dim - 1]  # MATLAB 1-based
        return self._A.shape
    
    def nnz(self) -> int:
        """Number of non-zero elements."""
        if self._A is None:
            return 0
        return np.count_nonzero(self._A)
    
    def svd(self) -> np.ndarray:
        """Singular values of the network matrix."""
        if self._A is None:
            raise ValueError("Network matrix not set")
        return np.linalg.svd(self._A.astype(float), compute_uv=False)
    
    def __matmul__(self, p: np.ndarray) -> np.ndarray:
        """Matrix multiplication for perturbation response: net @ p"""
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return self._G @ p
    
    def populate(self, source: Union['Network', np.ndarray, Dict]):
        """Populate from another Network, matrix, or dict."""
        if isinstance(source, Network):
            self._A = source._A.copy() if source._A is not None else None
            self._G = source._G.copy() if source._G is not None else None
            self._network = source._network
            self.names = source.names.copy()
            self.description = source.description
            self.created = source.created.copy()
        elif isinstance(source, np.ndarray):
            self.setA(source)
        elif isinstance(source, dict):
            for key, value in source.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        else:
            raise ValueError("Source must be Network, ndarray, or dict")
    
    def save(self, *args):
        """Save the network (calls parent save)."""
        super().save(*args)
    
    @staticmethod
    def load(*args):
        """Load network (calls parent load)."""
        return Exchange.load(*args)
    
    @staticmethod
    def fetch(url_or_name: Optional[str] = None, **kwargs):
        """Fetch network from URL or repository.
        
        Args:
            url_or_name: URL or network name
            **kwargs: Options like baseurl, version, type, N, etc.
        """
        options = {
            'directurl': '',
            'baseurl': 'https://bitbucket.org/sonnhammergrni/gs-networks/raw/',
            'version': 'master',
            'type': 'random',
            'N': 10,
            'name': '',
            'filelist': False,
            'filetype': ''
        }
        options.update(kwargs)
        
        if url_or_name is None:
            # Default case
            default_file = 'Nordling-D20100302-random-N10-L25-ID1446937.json'
            obj_data = Exchange.fetch(options, default_file)
        else:
            obj_data = Exchange.fetch(options, url_or_name)
        
        if isinstance(obj_data, list):
            return obj_data
        else:
            net = Network()
            net.populate(obj_data)
            return net
