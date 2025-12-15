import numpy as np
import os
from datetime import datetime
from math import floor
import requests
from typing import Optional, Dict, Any, List, Union
from .Exchange import Exchange

class Network(Exchange):
    """Stores a network matrix A and calculates important network properties."""
    
    def __init__(self, A: Optional[np.ndarray] = None, network_type: str = 'unknown') -> None:
        super().__init__()
        self._A: Optional[np.ndarray] = None
        self._G: Optional[np.ndarray] = None  # Static gain model
        self._network: str = ''
        self._names: List[str] = []
        self.description: str = ''
        self.tol: float = np.finfo(float).eps
        self.created: Dict[str, Any] = {
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
    
    def setA(self, A: Union[np.ndarray, List[Any]]) -> None:
        """Set the adjacency matrix and compute derived properties."""
        if not isinstance(A, np.ndarray):
            A = np.array(A)
        self._A = A
        # Calculate G = -pinv(A)
        # Use float type for calculation
        A_float = A.astype(float)
        self._G = -np.linalg.pinv(A_float)
        
        cond_val = np.linalg.cond(A_float)
        if np.any(np.isinf(cond_val)) or np.any(np.isnan(cond_val)):
            id_val = 'inf'
        else:
            if isinstance(cond_val, np.ndarray):
                val = np.mean(cond_val)
            else:
                val = float(cond_val)
            id_val = str(round(val * 10000))
            
        self.created['id'] = id_val
        self.created['nodes'] = str(A.shape[0])
        self.created['sparsity'] = str(np.count_nonzero(A))
    
    @classmethod
    def from_json_url(cls, url: str) -> 'Network':
        """Create a Network instance from a JSON file at the given URL."""
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'obj_data' in data:
            obj = data['obj_data']
        else:
            obj = data

        network = cls()
        if 'A' in obj:
            network.setA(np.array(obj['A']))
        
        if 'names' in obj:
            network.names = obj['names']
            
        if 'network' in obj:
            network.network = obj['network']
            
        return network

    def setname(self, namestruct: Optional[Dict[str, Any]] = None) -> None:
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
        # Handle potential missing keys or non-string values safely
        creator = namer.get('creator', '')
        net_type = namer.get('type', 'unknown')
        nodes = namer.get('nodes', '0')
        # Recalculate sparsity if A is set, else use stored
        links = np.count_nonzero(self._A) if self._A is not None else 0
        net_id = namer.get('id', '')
        
        time_str = datetime.now().strftime('%Y%m%d')
        self._network = f"{creator}-D{time_str}-{net_type}-N{nodes}-L{links}-ID{net_id}"
    
    @property
    def A(self) -> Optional[np.ndarray]:
        return self._A
    
    @A.setter
    def A(self, value: np.ndarray) -> None:
        self.setA(value)
    
    @property
    def G(self) -> Optional[np.ndarray]:
        return self._G
    
    @property
    def network(self) -> str:
        return self._network
    
    @network.setter
    def network(self, value: str) -> None:
        self._network = value
    
    @property
    def N(self) -> int:
        """Number of nodes."""
        return self._A.shape[0] if self._A is not None else 0
    
    @property
    def names(self) -> List[str]:
        """Node names, generates defaults if empty."""
        if not self._names and self.N > 0:
            # Generate default names G01, G02, ...
            digits = floor(np.log10(self.N)) + 1 if self.N > 0 else 1
            self._names = [f"G{i+1:0{digits}d}" for i in range(self.N)]
        return self._names
    
    @names.setter
    def names(self, value: List[str]) -> None:
        self._names = value
    
    def show(self) -> None:
        """Display network matrix and properties (text-based approximation)."""
        if self._A is None:
            print("No network matrix to display")
            return
        
        print("Network Matrix:")
        print(self._A)
        print("\nNetwork Properties:")
        print(f"Name: {self.network}")
        print(f"Description: {self.description}")
        if self._A.size > 0:
            print(f"Sparseness: {np.count_nonzero(self._A) / self._A.size:.4f}")
        print(f"# Nodes: {self._A.shape[0]}")
        print(f"# Links: {np.count_nonzero(self._A)}")
    
    def view(self) -> None:
        """Graphical network view (placeholder)."""
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
            # MATLAB 1-based indexing support
            if dim < 1 or dim > self._A.ndim:
                 raise ValueError(f"Dimension {dim} out of bounds")
            return self._A.shape[dim - 1]
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
        if self._G is None:
             raise ValueError("G matrix not calculated")
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return self._G @ p
    
    def populate(self, source: Union['Network', np.ndarray, Dict[str, Any]]) -> None:
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
                # Also check for private attributes if key matches
                elif hasattr(self, f"_{key}"):
                    setattr(self, f"_{key}", value)
        else:
            raise ValueError("Source must be Network, ndarray, or dict")
    
    def save(self, *args: Any) -> None:
        """Save the network (calls parent save)."""
        super().save(*args)
    
    @staticmethod
    def load(*args: Any) -> Any:
        """Load network (calls parent load)."""
        return Exchange.load(*args)
    
    @staticmethod
    def fetch(url_or_name: Optional[str] = None, **kwargs: Any) -> Any:
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
        
        # Unwrap obj_data if present
        if isinstance(obj_data, dict) and 'obj_data' in obj_data:
            obj_data = obj_data['obj_data']
        
        if isinstance(obj_data, list):
            return obj_data
        else:
            net = Network()
            net.populate(obj_data)
            return net
