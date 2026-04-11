import numpy as np
from typing import Optional
from .Exchange import Exchange
from .Network import Network


class Experiment(Exchange):
    """Generates experimental data for a Network."""

    def __init__(self, network: Optional[Network] = None, scale: float = 1.0) -> None:
        super().__init__()
        self._G: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._E: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._scale = scale

        if network:
            self.populate(network)
            self.gaussian()

    def populate(self, source: object) -> None:
        """Initialize from a Network."""
        if not isinstance(source, Network):
            raise TypeError("Experiment.populate requires a Network")
        if source.A is None:
            raise ValueError("Network A matrix must be set")
        self._G = source.G if source.G is not None else np.eye(source.A.shape[0])
        source_P = getattr(source, "P", None)
        self._P = source_P if source_P is not None else np.eye(source.A.shape[0])

    def gaussian(self) -> "Experiment":
        """Generate Gaussian noise and response."""
        if self._P is None or self._G is None:
            raise ValueError("P and G must be set before generating noise")
        n_rows, n_cols = self._G.shape[0], self._P.shape[1]
        self._E = self._scale * np.random.randn(n_rows, n_cols)
        self._Y = self.signal() + self.noise()
        return self

    def signal(self) -> np.ndarray:
        """Compute true signal (G @ P)."""
        if self._G is None or self._P is None:
            raise ValueError("G and P must be set")
        return self._G @ self._P  # type: ignore[no-any-return]

    def noise(self) -> np.ndarray:
        """Return noise matrix."""
        if self._E is None:
            raise ValueError("Noise must be generated first")
        return self._E

    def noiseY(self) -> np.ndarray:
        """Return noisy response, generating if needed."""
        if self._Y is None:
            self.gaussian()
        if self._Y is None:
            raise ValueError("Failed to generate Y")
        return self._Y

    def trueY(self) -> np.ndarray:
        """Return true response without noise."""
        if self._Y is None:
            self.gaussian()
        return self.signal()

    @property
    def G(self) -> Optional[np.ndarray]:
        return self._G

    @property
    def P(self) -> Optional[np.ndarray]:
        return self._P

    @property
    def E(self) -> Optional[np.ndarray]:
        return self._E

    @property
    def Y(self) -> Optional[np.ndarray]:
        return self._Y
