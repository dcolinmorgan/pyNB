from datetime import datetime
from typing import Any

import numpy as np
from numpy import linalg

from .Exchange import Exchange
from .Experiment import Experiment
from .Network import Network


class Dataset(Exchange):
    """Stores data complementary to a Network, with minimal computation."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        # Core attributes
        self._network: Network | None = None  # Network object
        self._Y: np.ndarray | None = None  # Observed expression response
        self._E: np.ndarray | None = None  # Expression noise
        self._P: np.ndarray | None = None  # Perturbations
        self._lambda: float | np.ndarray | None = None  # Noise variance
        self._names: list[str] | None = None  # Gene names
        self._created: dict[str, Any] = {
            "creator": "dc",  # os.getlogin(),
            "time": datetime.now(),
            "id": "",
        }
        self._dataset_name: str = ""

        # Populate from arguments
        if args:
            self.populate(args[0])
        self._set_name()

    def populate(self, source: Any) -> None:
        """Populate data from Network or Experiment."""
        if isinstance(source, Network):
            self._network = source
            source_P = getattr(source, "P", None)
            if source.A is not None:
                self._P = (
                    source_P if source_P is not None else np.eye(source.A.shape[0])
                )
                self._Y = source.A @ self._P
            else:
                self._P = source_P
                self._Y = None

        elif isinstance(source, Experiment):
            self._network = Network(source._G)
            self._Y = source.trueY()
            self._P = source._P
            self._E = source._E
            self._lambda = np.var(source._E) if source._E is not None else 1.0

        self._created["id"] = (
            str(round(linalg.cond(self._Y) * 10000)) if self._Y is not None else ""
        )

    def true_response(self) -> np.ndarray | None:
        """Compute true response (G @ P)."""
        if self._network is None:
            if self._Y is not None:
                self._network = Network(np.eye(self._Y.shape[0]))
            else:
                return None

        if self._network.G is None and self._network.A is None:
            return None

        G = self._network.G if self._network.G is not None else self._network.A

        if self._P is not None and G is not None:
            result: np.ndarray = G @ self._P
            return result
        return G

    def _set_name(self) -> None:
        """Generate dataset identifier."""
        creator = self._created["creator"]
        time_str = self._created["time"].strftime("%Y%m%d")

        net_id = ""
        if self._network and self._network.network and "-ID" in self._network.network:
            net_id = self._network.network.split("-ID")[-1]

        n = self._P.shape[0] if self._P is not None else 0
        e = self._P.shape[1] if self._P is not None else 0

        self._dataset_name = (
            f"{creator}-ID{net_id}-D{time_str}-N{n}-E{e}-IDY{self._created['id']}"
        )

    # Properties
    @property
    def dataset(self) -> str:
        return self._dataset_name

    @property
    def network(self) -> Network | None:
        return self._network

    @network.setter
    def network(self, value: Network) -> None:
        self._network = value

    @property
    def Y(self) -> np.ndarray | None:
        return self._Y

    @Y.setter
    def Y(self, value: np.ndarray) -> None:
        self._Y = value

    @property
    def E(self) -> np.ndarray | None:
        return self._E

    @E.setter
    def E(self, value: np.ndarray) -> None:
        self._E = value

    @property
    def P(self) -> np.ndarray | None:
        return self._P

    @P.setter
    def P(self, value: np.ndarray) -> None:
        self._P = value

    @property
    def lambda_(self) -> float | np.ndarray | None:
        return self._lambda

    @lambda_.setter
    def lambda_(self, value: float | np.ndarray) -> None:
        self._lambda = value

    @property
    def N(self) -> int:
        """Number of genes (rows in Y matrix)."""
        return self._Y.shape[0] if self._Y is not None else 0

    @property
    def M(self) -> int:
        """Number of samples (columns in Y matrix)."""
        return self._Y.shape[1] if self._Y is not None else 0

    @property
    def gene_names(self) -> list[str] | None:
        """Gene names."""
        if self._names is None and self.N > 0:
            return [f"G{i + 1}" for i in range(self.N)]
        return self._names

    @gene_names.setter
    def gene_names(self, value: list[str]) -> None:
        self._names = value
