import numpy as np
from numpy import linalg
from scipy import sparse
from datastruct.Network import Network
from analyze.DataModel import DataModel
import networkx as nx

class Model(DataModel):
    """Calculates properties related to the supplied network."""

    def __init__(self, mat, **kwargs):
        super().__init__()
        # Private properties
        self._network = ''           # Identifier for network
        self._interampatteness = None  # Condition number
        self._proximity_ratio = None   # Small-world tendency
        self._networkComponents = None # Number of strong components
        self._medianPathLength = None  # Median path length
        self._meanPathLength = None    # Mean path length
        self._tauG = None             # Time constant
        self._CC = None               # Average clustering coefficient
        self._DD = None               # Average degree distribution

        # Set tolerance if provided
        if 'tol' in kwargs:
            self.tol(kwargs['tol'])

        # Analyze the model
        self.analyse_model(mat, **kwargs)

    def analyse_model(self, mat, **kwargs):
        """Perform analysis on the provided network."""
        self._network = self.identifier(mat)
        self._interampatteness = self.cond(mat)
        self._networkComponents = self.graphconncomp(mat)
        self._medianPathLength, self._meanPathLength = self.median_path_length(mat)
        self._tauG = self.time_constant(mat)
        self._CC = np.nanmean(self.clustering_coefficient(mat))
        self._DD = np.mean(self.degree_distribution(mat))
        self._proximity_ratio = self.calc_proximity_ratio(mat)

    # Property getters
    @property
    def network(self):
        return self._network

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

    # Static methods
    @classmethod
    def give_matrix(cls, model, inv=None):
        """Return the network matrix or its inverse."""
        if isinstance(model, Network):
            return model.G if inv == 'inv' else model.A
        return model

    @classmethod
    def identifier(cls, model):
        """Get network identifier."""
        return model.network if isinstance(model, Network) else ''

    @classmethod
    def cond(cls, model):
        """Calculate condition number (interampatteness)."""
        A = cls.give_matrix(model)
        return linalg.cond(A)

    @classmethod
    def median_path_length(cls, model, **kwargs):
        """Calculate median and mean path lengths."""
        A = cls.give_matrix(model)
        directed = cls.type() == 'directed'
        G = nx.from_numpy_array(A, create_using=nx.DiGraph if directed else nx.Graph)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        pl = []
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j and j in lengths[i]:
                    pl.append(lengths[i][j])
        pl = np.array(pl)
        median_pl = np.median(pl) if pl.size > 0 else np.inf
        mean_pl = np.mean(pl) if pl.size > 0 else np.inf
        return median_pl, mean_pl

    @classmethod
    def graphconncomp(cls, model):
        """Calculate number of strongly connected components."""
        A = cls.give_matrix(model)
        directed = cls.type() == 'directed'
        G = nx.from_numpy_array(np.sign(A), create_using=nx.DiGraph if directed else nx.Graph)
        return nx.number_strongly_connected_components(G) if directed else nx.number_connected_components(G)

    @classmethod
    def time_constant(cls, model):
        """Calculate minimum time constant of the model G."""
        G = cls.give_matrix(model, 'inv')
        eigenvalues = linalg.eigvals(G)
        tauG = np.min(1 / np.abs(np.real(eigenvalues)))
        return tauG

    @classmethod
    def clustering_coefficient(cls, model):
        """Calculate clustering coefficient."""
        A = cls.give_matrix(model)
        A = A.copy()
        np.fill_diagonal(A, 0)  # Remove self-loops
        directed = cls.type() == 'directed'
        
        if directed:
            C_out = np.zeros(A.shape[0])
            C_in = np.zeros(A.shape[0])
            for i in range(A.shape[0]):
                out_vertices = np.where(A[:, i])[0]
                in_vertices = np.where(A[i, :])[0]
                subnet_out = A[out_vertices][:, out_vertices]
                subnet_in = A[in_vertices][:, in_vertices]
                C_out[i] = np.sum(subnet_out) / (subnet_out.size - len(out_vertices)) if len(out_vertices) > 1 else 0
                C_in[i] = np.sum(subnet_in) / (subnet_in.size - len(in_vertices)) if len(in_vertices) > 1 else 0
            return C_out  # Only returning C_out to match MATLAB’s single-output default
        else:
            G = nx.from_numpy_array(A + A.T, create_using=nx.Graph)
            C = list(nx.clustering(G).values())
            return np.array(C)

    @classmethod
    def degree_distribution(cls, model):
        """Calculate degree distribution."""
        A = cls.give_matrix(model)
        A = A.copy()
        np.fill_diagonal(A, 0)  # Remove self-loops
        A = A.astype(bool)
        
        if cls.type() == 'directed':
            out_degree = np.sum(A, axis=1)
            return out_degree  # Only returning out_degree to match MATLAB’s single-output default
        else:
            A = A | A.T
            degree_dist = np.sum(A, axis=1)
            return degree_dist

    @classmethod
    def calc_proximity_ratio(cls, model):
        """Calculate proximity ratio (small-worldness)."""
        A = cls.give_matrix(model)
        A = A.copy()
        np.fill_diagonal(A, 0)  # Remove self-loops
        A = A.astype(bool)
        N = A.shape[0]

        _, L = cls.median_path_length(A)
        md = np.sum(A) / N
        Lr = np.log(N) / np.log(md) if md > 1 else np.inf
        C = np.mean(cls.clustering_coefficient(A))
        Cr = np.sum(A) / (N * (N - 1))
        smallworldness = (C / Cr) * (Lr / L) if L > 0 and Cr > 0 else np.inf
        return smallworldness
