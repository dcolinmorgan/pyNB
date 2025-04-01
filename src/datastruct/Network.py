import numpy as np
from numpy import linalg
from datetime import datetime
import os
from datastruct.Exchange import Exchange

class Network(Exchange):
    """Stores a network matrix A and calculates network properties."""

    def __init__(self, *args):
        super().__init__()
        # Private properties
        self._network = None  # Unique name of network
        self._A = None        # Network matrix
        self._G = None        # Static gain model

        # Public properties
        self._names = []
        self.description = ''

        # Hidden properties
        self.created = {
            'creator': os.getlogin(),
            'time': datetime.now(),
            'id': '',
            'nodes': '',
            'type': 'unknown',
            'sparsity': ''
        }
        self.tol = np.finfo(float).eps
        self._N = None  # Number of nodes

        # Handle input arguments
        if len(args) == 2:
            self.setA(args[0])
            self.created['type'] = args[1]
            self.setname()
        elif len(args) == 1:
            self.setA(args[0])
            self.setname()

    # Getter and Setter for network properties
    @property
    def network(self):
        return self._network

    @property
    def A(self):
        return self._A

    @property
    def G(self):
        return self._G

    def setA(self, A):
        """Set the network matrix A and compute G."""
        self._A = np.asarray(A)
        self._G = -linalg.pinv(self._A)  # Full matrix pseudo-inverse
        self.created['id'] = str(round(linalg.cond(self._A) * 10000))
        self.created['nodes'] = str(self._A.shape[0])
        self.created['sparsity'] = str(self.nnz())
        self._network = self.network  # Trigger name update if needed

    def setname(self, namestruct=None):
        """Set the network name based on creation metadata."""
        if namestruct is None:
            namestruct = self.created
        if not isinstance(namestruct, dict):
            raise ValueError("Input argument must be name/value pairs in dict form")

        namer = self.created.copy()
        for key, value in namestruct.items():
            if key in namer:
                namer[key] = value

        self._network = (f"{namer['creator']}-D{namer['time'].strftime('%Y%m%d')}-"
                         f"{namer['type']}-N{namer['nodes']}-L{self.nnz()}-ID{namer['id']}")

    @property
    def N(self):
        """Number of nodes in the network."""
        return self.size()[0] if self._A is not None else 0

    @property
    def names(self):
        """Get or generate node names."""
        if not self._names:
            return [f"G{i:0{int(np.log10(self.N)) + 1}d}" for i in range(1, self.N + 1)]
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    def show(self):
        """Display network matrix and properties (requires GUI library)."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.table import Table

            # Matrix display
            fig1, ax1 = plt.subplots()
            ax1.matshow(self._A, cmap='viridis')
            ax1.set_xticks(range(self.N))
            ax1.set_yticks(range(self.N))
            ax1.set_xticklabels(self.names)
            ax1.set_yticklabels(self.names)
            plt.title("Network Matrix")

            # Properties table
            network_properties = [
                ('Name', self.network),
                ('Description', self.description),
                ('Sparseness', self.nnz() / self._A.size),
                ('# Nodes', self._A.shape[0]),
                ('# Links', self.nnz())
            ]
            fig2, ax2 = plt.subplots()
            ax2.axis('off')
            table = Table(ax2, bbox=[0, 0, 1, 1])
            for i, (prop, val) in enumerate(network_properties):
                table.add_cell(i, 0, 0.3, 0.1, text=prop, loc='center')
                table.add_cell(i, 1, 0.7, 0.1, text=str(val), loc='center')
            ax2.add_table(table)
            plt.title("Network Properties")

            plt.show()
        except ImportError:
            print("Visualization requires matplotlib. Install it with 'pip install matplotlib'.")

    def view(self):
        """Rough graphical network plot (requires networkx)."""
        try:
            import networkx as nx
            import matplotlib.pyplot as plt

            Ap = self._A.copy()
            np.fill_diagonal(Ap, 0)  # Remove self-loops
            G = nx.DiGraph(Ap)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, labels={i: name for i, name in enumerate(self.names)})
            edge_labels = {(i, j): f"{Ap[i, j]:.2f}" for i, j in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            plt.show()
            return G
        except ImportError:
            print("Network visualization requires networkx. Install it with 'pip install networkx'.")
            return None

    def sign(self):
        """Return the sign of the network matrix."""
        return np.sign(self._A)

    def logical(self):
        """Return a logical (boolean) version of the network matrix."""
        return self._A.astype(bool)

    def size(self, dim=None):
        """Return the size of the network matrix."""
        if dim is not None:
            return self._A.shape[dim - 1]  # MATLAB uses 1-based indexing
        return self._A.shape

    def nnz(self):
        """Return the number of non-zero elements in the network matrix."""
        return np.count_nonzero(self._A)

    def svd(self):
        """Return the singular values of A."""
        return linalg.svd(self._A, compute_uv=False)

    def __mul__(self, p):
        """Generate a steady-state response to a perturbation."""
        p = np.atleast_2d(p)
        if p.shape[0] == 1:
            p = p.T
        return self._G @ p

    def populate(self, input):
        """Populate the Network object with fields from input."""
        if not isinstance(input, (dict, Network)):
            raise ValueError("Input must be a dict or Network object")
        super().populate(input)
        if 'A' in (input if isinstance(input, dict) else vars(input)):
            self.setA(self._A)  # Ensure G and metadata are updated
        return self

    def save(self, *args, **kwargs):
        """Save the network to a file."""
        super().save(*args, **kwargs)

    @staticmethod
    def load(*args):
        """Load a network from a file."""
        return Exchange.load(*args)

    @staticmethod
    def fetch(*args):
        """Fetch a network from a remote repository."""
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
        if not args:
            default_file = 'Nordling-D20100302-random-N10-L25-ID1446937.json'
            obj_data = Exchange.fetch(options, default_file)
        else:
            obj_data = Exchange.fetch(options, *args)

        if isinstance(obj_data, list):
            return obj_data
        net = Network()
        net.populate(obj_data)
        return net
