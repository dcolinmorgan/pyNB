import numpy as np
from numpy import linalg
import os
from datetime import datetime
from datastruct.Exchange import Exchange
from datastruct.Network import Network
from datastruct.Experiment import Experiment

class Dataset(Exchange):
    """Stores data complementary to a Network, with minimal computation."""
    
    def __init__(self, *args):
        super().__init__()
        # Core attributes
        self._network = None  # Network object
        self._Y = None        # Observed expression response
        self._E = None        # Expression noise
        self._P = None        # Perturbations
        self._lambda = None   # Noise variance
        self._names = None    # Gene names
        self._created = {
            'creator': 'dc', #os.getlogin(),
            'time': datetime.now(),
            'id': ''
        }
        self._dataset_name = ''
        
        # Populate from arguments
        if args:
            self.populate(args[0])
        self._set_name()

    def populate(self, source):
        """Populate data from Network or Experiment."""
        if isinstance(source, Network):
            self._network = source
            self._P = source.P if source.P is not None else np.eye(source.A.shape[0])
            self._Y = source.A @ self._P
        elif isinstance(source, Experiment):
            self._network = Network(source._G)
            self._Y = source.trueY()
            self._P = source._P
            self._E = source._E
            self._lambda = np.var(source._E) if source._E is not None else 1.0
        self._created['id'] = str(round(linalg.cond(self._Y) * 10000)) if self._Y is not None else ''

    def true_response(self):
        """Compute true response (G @ P)."""
        if self._network is None or self._network.G is None:
            self._network = Network(np.eye(self._Y.shape[0]) if self._Y is not None else np.eye(1))
        G = self._network.G if self._network.G is not None else self._network.A
        return G @ self._P if self._P is not None else G

    def _set_name(self):
        """Generate dataset identifier."""
        creator = self._created['creator']
        time_str = self._created['time'].strftime('%Y%m%d')
        net_id = self._network.network.split('-ID')[-1] if self._network and '-ID' in self._network.network else ''
        n = self._P.shape[0] if self._P is not None else 0
        e = self._P.shape[1] if self._P is not None else 0
        self._dataset_name = f"{creator}-ID{net_id}-D{time_str}-N{n}-E{e}-IDY{self._created['id']}"

    # Properties
    @property
    def dataset(self):
        return self._dataset_name

    @property
    def network(self):
        return self._network

    @property
    def Y(self):
        return self._Y

    @property
    def E(self):
        return self._E

    @property
    def P(self):
        return self._P

    @property
    def lambda_(self):
        return self._lambda

    @property
    def N(self):
        """Number of genes (rows in Y matrix)."""
        return self._Y.shape[0] if self._Y is not None else 0

    @property
    def M(self):
        """Number of samples (columns in Y matrix)."""
        return self._Y.shape[1] if self._Y is not None else 0

    @property
    def gene_names(self):
        """Gene names."""
        if self._names is None and self.N > 0:
            return [f"G{i+1}" for i in range(self.N)]
        return self._names
