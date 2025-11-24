import numpy as np
import os
from datetime import datetime
import requests
import json
from typing import Optional, Dict, Any
from datastruct.Exchange import Exchange

class Network(Exchange):
    """Represents a network structure."""
    
    def __init__(self, A: np.ndarray = None):
        super().__init__()
        self._A = A             # Adjacency matrix
        self._P = None          # Perturbations
        self._G = None          # Static gain model
        self._network_id = self._generate_id() if A is not None else ''

    @classmethod
    def from_json_url(cls, url: str) -> 'Network':
        """Create a Network instance from a JSON file at the given URL.
        
        Args:
            url: URL to the JSON file containing network data
            
        Returns:
            Network instance initialized with the adjacency matrix from the JSON data
            
        Raises:
            requests.exceptions.RequestException: If the URL request fails
            ValueError: If the JSON data is invalid or doesn't contain the adjacency matrix
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            
            if 'obj_data' not in data:
                raise ValueError("JSON data does not contain 'obj_data' field")
                
            obj_data = data['obj_data']
            if 'A' not in obj_data:
                raise ValueError("JSON data does not contain adjacency matrix in obj_data.A")
                
            adjacency_data = obj_data['A']
            if not isinstance(adjacency_data, list) or not all(
                isinstance(row, list) for row in adjacency_data
            ):
                raise ValueError("Adjacency matrix data must be a 2D array")
                
            A = np.array(adjacency_data, dtype=float)
            network = cls(A)
            
            # Set network ID if available
            if 'network' in obj_data:
                network.network = obj_data['network']
                
            return network
            
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to fetch network data from URL: {e}"
            )
        except ValueError as e:
            raise ValueError(f"Invalid network data format: {e}")

    @classmethod
    def from_json_file(cls, file_path: str) -> 'Network':
        """Create a Network instance from a local JSON file.
        
        Args:
            file_path: Path to the JSON file containing network data
            
        Returns:
            Network instance initialized with the adjacency matrix from the JSON data
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the JSON data is invalid or doesn't contain the adjacency matrix
        """
        try:
            with open(file_path, 'r') as f:
                data: Dict[str, Any] = json.load(f)
            
            if 'obj_data' not in data:
                raise ValueError("JSON data does not contain 'obj_data' field")
                
            obj_data = data['obj_data']
            if 'A' not in obj_data:
                raise ValueError("JSON data does not contain adjacency matrix in obj_data.A")
                
            adjacency_data = obj_data['A']
            if not isinstance(adjacency_data, list) or not all(
                isinstance(row, list) for row in adjacency_data
            ):
                raise ValueError("Adjacency matrix data must be a 2D array")
                
            A = np.array(adjacency_data, dtype=float)
            network = cls(A)
            
            # Set network ID if available
            if 'network' in obj_data:
                network.network = obj_data['network']
                
            return network
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Network file not found: {file_path}")
        except ValueError as e:
            raise ValueError(f"Invalid network data format: {e}")

    def populate(self, source):
        """Populate from another Network or matrix."""
        if isinstance(source, Network):
            self._A = source.A
            self._P = source.P
            self._G = source.G
            self._network_id = source.network
        elif isinstance(source, np.ndarray):
            self._A = source
            self._network_id = self._generate_id()
        else:
            raise ValueError("Source must be a Network or NumPy array")

    def _generate_id(self):
        """Generate unique network identifier."""
        N = self._A.shape[0] if self._A is not None else 0
        L = np.sum(self._A != 0) if self._A is not None else 0
        ID = str(round(np.random.rand() * 10000))
        return f"dc-D{datetime.now().strftime('%Y%m%d')}-directed-N{N}-L{L}-ID{ID}"

    # Properties
    @property
    def A(self):
        return self._A

    @property
    def P(self):
        return self._P

    @property
    def G(self):
        return self._G

    @property
    def network(self):
        return self._network_id

    @network.setter
    def network(self, value):
        self._network_id = value
