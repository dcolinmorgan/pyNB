import numpy as np
from datastruct.Exchange import Exchange
from datastruct.Network import Network

class Experiment(Exchange):
    """Generates experimental data for a Network."""
    
    def __init__(self, network: Network = None, scale: float = 1.0):
        super().__init__()
        self._G = None       # Static gain model
        self._P = None       # Perturbations
        self._E = None       # Noise
        self._Y = None       # Response with noise
        self._scale = scale  # Noise scale
        
        if network:
            self.populate(network)
            self.gaussian()

    def populate(self, network: Network):
        """Initialize from a Network."""
        if network.A is None:
            raise ValueError("Network A matrix must be set")
        self._G = network.G if network.G is not None else np.eye(network.A.shape[0])
        self._P = network.P if network.P is not None else np.eye(network.A.shape[0])

    def gaussian(self):
        """Generate Gaussian noise and response."""
        if self._P is None or self._G is None:
            raise ValueError("P and G must be set before generating noise")
        n_rows, n_cols = self._G.shape[0], self._P.shape[1]
        self._E = self._scale * np.random.randn(n_rows, n_cols)
        self._Y = self.signal() + self.noise()
        return self

    def signal(self):
        """Compute true signal (G @ P)."""
        return self._G @ self._P

    def noise(self):
        """Return noise matrix."""
        if self._E is None:
            raise ValueError("Noise must be generated first")
        return self._E

    def noiseY(self):
        """Return noisy response, generating if needed."""
        if self._Y is None:
            self.gaussian()
        return self._Y

    def trueY(self):
        """Return true response without noise."""
        if self._Y is None:
            self.gaussian()
        return self.signal()

    # Properties
    @property
    def G(self):
        return self._G

    @property
    def P(self):
        return self._P

    @property
    def E(self):
        return self._E

    @property
    def Y(self):
        return self._Y
