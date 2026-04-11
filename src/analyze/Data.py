from typing import Any

import numpy as np
import requests
from numpy import linalg
from scipy.stats import chi2

from datastruct.Dataset import Dataset
from datastruct.Network import Network

from .DataModel import DataModel


class Data(DataModel):
    """Analyzes properties of a Dataset."""

    def __init__(self, dataset: Dataset, tol: float | None = None) -> None:
        super().__init__(dataset)
        self._dataset_id: str = dataset.dataset
        self._tol: float = tol if tol is not None else float(np.finfo(float).eps)
        self._SNR_Phi_true: float = 0.0
        self._SNR_Phi_gauss: float = 0.0
        self._SNR_L: float = 0.0
        self._SNR_phi_true: float = 0.0
        self._SNR_phi_gauss: float = 0.0
        self._analyze()

    @classmethod
    def from_json_url(cls, url: str) -> "Data":
        """Create a Data instance from a JSON file at the given URL."""
        response = requests.get(url)
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        if "obj_data" not in data:
            raise ValueError("JSON data does not contain 'obj_data' field")

        obj_data = data["obj_data"]
        dataset = cls._build_dataset(obj_data)
        return cls(dataset)

    @classmethod
    def from_json_file(cls, file_path: str) -> "Data":
        """Create a Data instance from a local JSON file."""
        import json

        with open(file_path) as f:
            data: dict[str, Any] = json.load(f)

        if "obj_data" not in data:
            raise ValueError("JSON data does not contain 'obj_data' field")

        obj_data = data["obj_data"]
        dataset = cls._build_dataset(obj_data)
        return cls(dataset)

    @classmethod
    def _build_dataset(cls, obj_data: dict[str, Any]) -> Dataset:
        """Build a Dataset from parsed JSON obj_data."""
        dataset = Dataset()

        if "dataset" in obj_data:
            dataset._dataset_name = obj_data["dataset"]

        if "network" in obj_data:
            network = Network()
            network.network = obj_data["network"]
            dataset._network = network

        for field in ["P", "E", "F", "Y", "cvP", "sdP", "svE", "sdY"]:
            if field in obj_data and obj_data[field]:
                setattr(dataset, f"_{field}", np.array(obj_data[field], dtype=float))

        for field in ["lambda", "SNR_L", "tol"]:
            if field in obj_data:
                setattr(dataset, f"_{field}", obj_data[field])

        if "names" in obj_data:
            dataset._names = obj_data["names"]
        if "created" in obj_data:
            dataset._created = obj_data["created"]

        return dataset

    def _analyze(self) -> None:
        """Compute all data properties."""
        ds = self._data
        if ds is None:
            return
        self._SNR_Phi_true = self._calc_SNR_Phi_true(ds)
        self._SNR_Phi_gauss = self._calc_SNR_Phi_gauss(ds)
        self._SNR_L = self._calc_SNR_L(ds)
        self._SNR_phi_true = float(np.min(self._calc_SNR_phi_true(ds)))
        self._SNR_phi_gauss = float(np.min(self._calc_SNR_phi_gauss(ds)))

    def _calc_SNR_Phi_true(self, ds: Dataset) -> float:
        """SNR: min(svd(true_response))/max(svd(E))."""
        true_resp = ds.true_response()
        if true_resp is None:
            return 0.0
        s_true = linalg.svd(true_resp, compute_uv=False)
        s_E = (
            linalg.svd(ds.E, compute_uv=False) if ds.E is not None else np.array([1.0])
        )
        return float(min(s_true) / max(s_E)) if s_E.size > 0 else float("inf")

    def _calc_SNR_Phi_gauss(self, ds: Dataset) -> float:
        """SNR with Gaussian assumption."""
        if ds.Y is None or ds.P is None:
            return float("inf")
        sigma = float(min(linalg.svd(ds.Y, compute_uv=False)))
        alpha = self.alpha() or 0.05
        lambda_val = self._get_lambda(ds)
        chi2_val = float(chi2.ppf(1 - alpha, ds.P.size))
        return float(sigma / np.sqrt(chi2_val * lambda_val))

    def _calc_SNR_L(self, ds: Dataset) -> float:
        """SNR: true expression to variance."""
        true_resp = ds.true_response()
        if true_resp is None or ds.P is None:
            return float("inf")
        sigma = float(min(linalg.svd(true_resp, compute_uv=False)))
        alpha = self.alpha() or 0.05
        lambda_val = self._get_lambda(ds)
        chi2_val = float(chi2.ppf(1 - alpha, ds.P.size))
        denom = np.sqrt(chi2_val * lambda_val)
        return float(sigma / denom) if denom != 0 else float("inf")

    def _calc_SNR_phi_true(self, ds: Dataset) -> np.ndarray:
        """Per-variable SNR (true)."""
        X = ds.true_response()
        if X is None:
            return np.array([0.0])
        return np.array(
            [
                float(linalg.norm(X[i, :]) / linalg.norm(ds.E[i, :]))
                if ds.E is not None and linalg.norm(ds.E[i, :]) > 0
                else float("inf")
                for i in range(X.shape[0])
            ]
        )

    def _calc_SNR_phi_gauss(self, ds: Dataset) -> np.ndarray:
        """Per-variable SNR (Gaussian)."""
        if ds.Y is None:
            return np.array([float("inf")])
        Y = ds.Y
        alpha = self.alpha() or 0.05
        lambda_val = self._get_lambda(ds)
        chi2_val = float(chi2.ppf(1 - alpha, Y.shape[1]))
        return np.array(
            [
                float(linalg.norm(Y[i, :])) / np.sqrt(chi2_val * lambda_val)
                for i in range(Y.shape[0])
            ]
        )

    @staticmethod
    def _get_lambda(ds: Dataset) -> float:
        """Extract lambda value from dataset."""
        if ds.lambda_ is None:
            return 1.0
        elif isinstance(ds.lambda_, (list, np.ndarray)):
            return float(np.mean(ds.lambda_))
        else:
            return float(ds.lambda_)

    @property
    def dataset(self) -> str:
        return self._dataset_id

    @property
    def SNR_Phi_true(self) -> float:
        return self._SNR_Phi_true

    @property
    def SNR_Phi_gauss(self) -> float:
        return self._SNR_Phi_gauss

    @property
    def SNR_L(self) -> float:
        return self._SNR_L

    @property
    def SNR_phi_true(self) -> float:
        return self._SNR_phi_true

    @property
    def SNR_phi_gauss(self) -> float:
        return self._SNR_phi_gauss
