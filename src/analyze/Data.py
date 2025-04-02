import numpy as np
from numpy import linalg
from scipy.stats import chi2
from datastruct.Dataset import Dataset
from analyze.DataModel import DataModel

class Data(DataModel):
    """Analyzes properties of a Dataset."""
    
    def __init__(self, dataset: Dataset, tol: float = None):
        super().__init__(dataset)
        self._dataset_id = dataset.dataset
        self._tol = tol if tol is not None else np.finfo(float).eps
        self._analyze()

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
        """SNR with Gaussian assumption."""
        sigma = min(linalg.svd(ds.Y, compute_uv=False))
        return sigma / np.sqrt(chi2.ppf(1 - self.alpha(), ds.P.size) * (ds.lambda_ or 1.0))

    def _calc_SNR_L(self, ds):
        """SNR: true expression to variance."""
        sigma = min(linalg.svd(ds.true_response(), compute_uv=False))
        denom = np.sqrt(chi2.ppf(1 - self.alpha(), ds.P.size) * (ds.lambda_ or 1.0))
        return sigma / denom if denom != 0 else float('inf')

    def _calc_SNR_phi_true(self, ds):
        """Per-variable SNR (true)."""
        X = ds.true_response()
        return np.array([
            linalg.norm(X[i, :]) / linalg.norm(ds.E[i, :]) if ds.E is not None and linalg.norm(ds.E[i, :]) > 0 else float('inf')
            for i in range(X.shape[0])
        ])

    def _calc_SNR_phi_gauss(self, ds):
        """Per-variable SNR (Gaussian)."""
        Y = ds.Y
        return np.array([
            linalg.norm(Y[i, :]) / np.sqrt(chi2.ppf(1 - self.alpha(), Y.shape[1]) * (ds.lambda_ or 1.0))
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
