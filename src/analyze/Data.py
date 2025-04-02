import numpy as np
from numpy import linalg
from scipy.stats import chi2
import requests
from typing import Dict, Any
from datastruct.Dataset import Dataset
from datastruct.Network import Network
from analyze.DataModel import DataModel

class Data(DataModel):
    """Analyzes properties of a Dataset."""
    
    def __init__(self, dataset: Dataset, tol: float = None):
        super().__init__(dataset)
        self._dataset_id = dataset.dataset
        self._tol = tol if tol is not None else np.finfo(float).eps
        self._analyze()

    @classmethod
    def from_json_url(cls, url: str) -> 'Data':
        """Create a Data instance from a JSON file at the given URL.
        
        Args:
            url: URL to the JSON file containing dataset data
            
        Returns:
            Data instance initialized with the dataset from the JSON data
            
        Raises:
            requests.exceptions.RequestException: If the URL request fails
            ValueError: If the JSON data is invalid or missing required fields
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            
            if 'obj_data' not in data:
                raise ValueError("JSON data does not contain 'obj_data' field")
                
            obj_data = data['obj_data']
            
            # Create Dataset instance
            dataset = Dataset()
            
            # Set basic attributes
            if 'dataset' in obj_data:
                dataset._dataset_name = obj_data['dataset']
            
            # Create and set Network
            if 'network' in obj_data:
                network = Network()
                network.network = obj_data['network']
                dataset._network = network
            
            # Set matrices
            for field in ['P', 'E', 'F', 'Y','cvP', 'sdP', 'svE', 'sdY']:
                if field in obj_data and obj_data[field]:
                    setattr(dataset, f'_{field}', np.array(obj_data[field], dtype=float))
            
            # Set scalar values
            for field in [ 'lambda', 'SNR_L', 'tol']:
                if field in obj_data:
                    setattr(dataset, f'_{field}', (obj_data[field]))
            
            # Set metadata
            if 'names' in obj_data:
                dataset._names = obj_data['names']
            if 'N' in obj_data:
                dataset._N = int(obj_data['N'])
            if 'M' in obj_data:
                dataset._M = int(obj_data['M'])
            if 'description' in obj_data:
                dataset._description = obj_data['description']
            if 'created' in obj_data:
                dataset._created = obj_data['created']
            
            return cls(dataset)
            
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(
                f"Failed to fetch dataset data from URL: {e}"
            )
        except ValueError as e:
            raise ValueError(f"Invalid dataset data format: {e}")

    def _analyze(self):
        """Compute all data properties."""
        ds = self._data
        self._SNR_Phi_true = self._calc_SNR_Phi_true(ds)
        self._SNR_Phi_gauss = self._calc_SNR_Phi_gauss(ds)
        self._SNR_L = self._calc_SNR_L(ds)
        self._SNR_phi_true = np.min(self._calc_SNR_phi_true(ds))
        self._SNR_phi_gauss = np.min(self._calc_SNR_phi_gauss(ds))

    def _calc_SNR_Phi_true(self, ds):
        """SNR: min(svd(true_response))/max(svd(E))."""
        s_true = linalg.svd(ds.true_response(), compute_uv=False)
        s_E = linalg.svd(ds.E, compute_uv=False) if ds.E is not None else np.array([1.0])
        return min(s_true) / max(s_E) if s_E.size > 0 else float('inf')

    def _calc_SNR_Phi_gauss(self, ds):
        """SNR with Gaussian assumption.
        
        Args:
            ds: Dataset object containing Y, P, and lambda values
            
        Returns:
            float: Signal-to-noise ratio under Gaussian assumption
        """
        if ds.Y is None or ds.P is None:
            return float('inf')
            
        sigma = min(linalg.svd(ds.Y, compute_uv=False))
        alpha = self.alpha()
        if alpha is None:
            alpha = 0.05  # Default significance level
            
        # Handle lambda_ which could be a list or single value
        if ds.lambda_ is None:
            lambda_val = 1.0
        elif isinstance(ds.lambda_, (list, np.ndarray)):
            lambda_val = float(np.mean(ds.lambda_))  # Take mean if it's a list/array
        else:
            lambda_val = float(ds.lambda_)  # Convert single value to float
            
        # Calculate chi2 quantile
        chi2_val = float(chi2.ppf(1 - alpha, ds.P.size))
        
        return sigma / np.sqrt(chi2_val * lambda_val)

    def _calc_SNR_L(self, ds):
        """SNR: true expression to variance.
        
        Args:
            ds: Dataset object containing true response, P, and lambda values
            
        Returns:
            float: Signal-to-noise ratio
        """
        if ds.true_response() is None or ds.P is None:
            return float('inf')
            
        sigma = min(linalg.svd(ds.true_response(), compute_uv=False))
        alpha = self.alpha()
        if alpha is None:
            alpha = 0.05  # Default significance level
            
        # Handle lambda_ which could be a list or single value
        if ds.lambda_ is None:
            lambda_val = 1.0
        elif isinstance(ds.lambda_, (list, np.ndarray)):
            lambda_val = float(np.mean(ds.lambda_))  # Take mean if it's a list/array
        else:
            lambda_val = float(ds.lambda_)  # Convert single value to float
            
        # Calculate chi2 quantile
        chi2_val = float(chi2.ppf(1 - alpha, ds.P.size))
        
        denom = np.sqrt(chi2_val * lambda_val)
        return sigma / denom if denom != 0 else float('inf')

    def _calc_SNR_phi_true(self, ds):
        """Per-variable SNR (true)."""
        X = ds.true_response()
        return np.array([
            linalg.norm(X[i, :]) / linalg.norm(ds.E[i, :]) if ds.E is not None and linalg.norm(ds.E[i, :]) > 0 else float('inf')
            for i in range(X.shape[0])
        ])

    def _calc_SNR_phi_gauss(self, ds):
        """Per-variable SNR (Gaussian).
        
        Args:
            ds: Dataset object containing Y and lambda values
            
        Returns:
            np.ndarray: Array of per-variable signal-to-noise ratios
        """
        if ds.Y is None:
            return np.array([float('inf')])
            
        Y = ds.Y
        alpha = self.alpha()
        if alpha is None:
            alpha = 0.05  # Default significance level
            
        # Handle lambda_ which could be a list or single value
        if ds.lambda_ is None:
            lambda_val = 1.0
        elif isinstance(ds.lambda_, (list, np.ndarray)):
            lambda_val = float(np.mean(ds.lambda_))  # Take mean if it's a list/array
        else:
            lambda_val = float(ds.lambda_)  # Convert single value to float
            
        # Calculate chi2 quantile
        chi2_val = float(chi2.ppf(1 - alpha, Y.shape[1]))
        
        return np.array([
            linalg.norm(Y[i, :]) / np.sqrt(chi2_val * lambda_val)
            for i in range(Y.shape[0])
        ])

    # Properties
    @property
    def dataset(self):
        return self._dataset_id

    @property
    def SNR_Phi_true(self):
        return self._SNR_Phi_true

    @property
    def SNR_Phi_gauss(self):
        return self._SNR_Phi_gauss

    @property
    def SNR_L(self):
        return self._SNR_L

    @property
    def SNR_phi_true(self):
        return self._SNR_phi_true

    @property
    def SNR_phi_gauss(self):
        return self._SNR_phi_gauss
