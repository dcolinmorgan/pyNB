import numpy as np
from numpy import linalg
import networkx as nx
from datastruct.Network import Network
from analyze.DataModel import DataModel

class Model(DataModel):
    """Analyzes structural properties of a Network."""
    
    def __init__(self, network: Network, tol: float = None):
        super().__init__(network)
        self._network_id = network.network
        self._tol = tol if tol is not None else np.finfo(float).eps
        self._analyze()

    def _analyze(self):
        """Compute all network properties."""
        net = self._data
        A = self._get_matrix(net)
        G = nx.from_numpy_array(np.sign(A), create_using=nx.DiGraph if self.type() == 'directed' else nx.Graph)
        
        self._interampatteness = linalg.cond(A)
        self._networkComponents = nx.number_strongly_connected_components(G) if self.type() == 'directed' else nx.number_connected_components(G)
        self._medianPathLength, self._meanPathLength = self._calc_path_lengths(G, A.shape[0])
        self._tauG = self._calc_time_constant(net)
        self._CC = np.nanmean(self._calc_clustering(A, G))
        self._DD = np.mean(self._calc_degree(A))
        self._proximity_ratio = self._calc_proximity_ratio(A, G)

    def _get_matrix(self, net, inv=False):
        """Extract matrix from Network."""
        return net.G if inv and net.G is not None else net.A

    def _calc_path_lengths(self, G, n):
        """Compute median and mean path lengths."""
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        pl = [lengths[i][j] for i in range(n) for j in range(n) if i != j and j in lengths[i]]
        return np.median(pl) if pl else np.inf, np.mean(pl) if pl else np.inf

    def _calc_time_constant(self, net):
        """Compute minimum time constant."""
        G = self._get_matrix(net, inv=True)
        eigenvalues = linalg.eigvals(G if G is not None else net.A)
        return np.min(1 / np.abs(np.real(eigenvalues)))

    def _calc_clustering(self, A, G):
        """Compute clustering coefficient."""
        A = A.copy()
        np.fill_diagonal(A, 0)
        if self.type() == 'directed':
            return np.array([
                np.sum(A[np.where(A[:, i])[0]][:, np.where(A[:, i])[0]]) / (np.sum(A[:, i]) * (np.sum(A[:, i]) - 1))
                if np.sum(A[:, i]) > 1 else 0
                for i in range(A.shape[0])
            ])
        return np.array(list(nx.clustering(G).values()))

    def _calc_degree(self, A):
        """Compute degree distribution."""
        A = A.copy()
        np.fill_diagonal(A, 0)
        A = A.astype(bool)
        return np.sum(A, axis=1) if self.type() == 'directed' else np.sum(A | A.T, axis=1)

    def _calc_proximity_ratio(self, A, G):
        """Compute small-worldness."""
        N = A.shape[0]
        _, L = self._calc_path_lengths(G, N)
        md = np.sum(A) / N
        Lr = np.log(N) / np.log(md) if md > 1 else np.inf
        C = np.mean(self._calc_clustering(A, G))
        Cr = np.sum(A) / (N * (N - 1))
        return (C / Cr) * (Lr / L) if L > 0 and Cr > 0 else np.inf

    # Properties
    @property
    def network(self):
        return self._network_id

    @property
    def interampatteness(self):
        return self._interampatteness

    @property
    def proximity_ratio(self):
        return self._proximity_ratio

    @property
    def networkComponents(self):
        return self._networkComponents

    @property
    def medianPathLength(self):
        return self._medianPathLength

    @property
    def meanPathLength(self):
        return self._meanPathLength

    @property
    def tauG(self):
        return self._tauG

    @property
    def CC(self):
        return self._CC

    @property
    def DD(self):
        return self._DD
