import numpy as np
from numpy import linalg
import networkx as nx
from typing import Optional, Tuple, Union, List
from datastruct.Network import Network
from analyze.DataModel import DataModel

class Model(DataModel):
    """Analyzes structural properties of a Network."""
    
    def __init__(self, network: Network, tol: Optional[float] = None) -> None:
        super().__init__(network)
        self._network_id: str = network.network
        self._tol: float = tol if tol is not None else np.finfo(float).eps
        
        # Initialize properties
        self._interampatteness: float = 0.0
        self._networkComponents: int = 0
        self._medianPathLength: float = 0.0
        self._meanPathLength: float = 0.0
        self._tauG: float = 0.0
        self._CC: float = 0.0
        self._DD: float = 0.0
        self._proximity_ratio: float = 0.0
        
        self._analyze()

    def _analyze(self) -> None:
        """Compute all network properties."""
        net = self._data
        if net is None or net.A is None:
            return
            
        A = self._get_matrix(net)
        # Create graph
        # Use abs(sign(A)) to get unweighted connections
        G = nx.from_numpy_array(np.abs(np.sign(A)), create_using=nx.DiGraph if self.type() == 'directed' else nx.Graph)
        
        self._interampatteness = linalg.cond(A)
        self._networkComponents = nx.number_strongly_connected_components(G) if self.type() == 'directed' else nx.number_connected_components(G)
        self._medianPathLength, self._meanPathLength = self._calc_path_lengths(G, A.shape[0])
        self._tauG = self._calc_time_constant(net)
        self._CC = float(np.nanmean(self._calc_clustering(A, G)))
        self._DD = float(np.mean(self._calc_degree(A)))
        self._proximity_ratio = self._calc_proximity_ratio(A, G)

    def _get_matrix(self, net: Network, inv: bool = False) -> np.ndarray:
        """Extract matrix from Network."""
        if inv and net.G is not None:
            return net.G
        return net.A if net.A is not None else np.array([])

    def _calc_path_lengths(self, G: nx.Graph, n: int) -> Tuple[float, float]:
        """Compute median and mean path lengths."""
        try:
            lengths = dict(nx.all_pairs_shortest_path_length(G))
            pl = [lengths[i][j] for i in range(n) for j in range(n) if i != j and j in lengths[i]]
            return float(np.median(pl)) if pl else np.inf, float(np.mean(pl)) if pl else np.inf
        except Exception:
            return np.inf, np.inf

    def _calc_time_constant(self, net: Network) -> float:
        """Compute minimum time constant."""
        G = self._get_matrix(net, inv=True)
        if G.size == 0:
            return 0.0
        try:
            eigenvalues = linalg.eigvals(G)
            # Avoid division by zero
            real_parts = np.abs(np.real(eigenvalues))
            real_parts = real_parts[real_parts > 1e-10]
            if len(real_parts) == 0:
                return np.inf
            return float(np.min(1 / real_parts))
        except linalg.LinAlgError:
            return np.inf

    def _calc_clustering(self, A: np.ndarray, G: nx.Graph) -> np.ndarray:
        """Compute clustering coefficient."""
        A = A.copy()
        np.fill_diagonal(A, 0)
        if self.type() == 'directed':
             # Custom directed clustering calculation from original code
             n = A.shape[0]
             res = []
             for i in range(n):
                 neighbors = np.where(A[:, i])[0]
                 k = np.sum(A[:, i])
                 if k > 1:
                     sub_A = A[neighbors][:, neighbors]
                     res.append(np.sum(sub_A) / (k * (k - 1)))
                 else:
                     res.append(0.0)
             return np.array(res)
        return np.array(list(nx.clustering(G).values()))

    def _calc_degree(self, A: np.ndarray) -> np.ndarray:
        """Compute degree distribution."""
        A = A.copy()
        np.fill_diagonal(A, 0)
        A = A.astype(bool)
        return np.sum(A, axis=1) if self.type() == 'directed' else np.sum(A | A.T, axis=1)

    def _calc_proximity_ratio(self, A: np.ndarray, G: nx.Graph) -> float:
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
    def network(self) -> str:
        return self._network_id

    @property
    def interampatteness(self) -> float:
        return self._interampatteness

    @property
    def proximity_ratio(self) -> float:
        return self._proximity_ratio

    @property
    def networkComponents(self) -> int:
        return self._networkComponents

    @property
    def medianPathLength(self) -> float:
        return self._medianPathLength

    @property
    def meanPathLength(self) -> float:
        return self._meanPathLength

    @property
    def tauG(self) -> float:
        return self._tauG

    @property
    def CC(self) -> float:
        return self._CC

    @property
    def DD(self) -> float:
        return self._DD
