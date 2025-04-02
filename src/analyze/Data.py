import numpy as np
from scipy import linalg
from scipy.stats import chi2
from datastruct.Dataset import Dataset
from datastruct.Network import Network
from analyze.DataModel import DataModel

class Data(DataModel):
    """Calculates data properties for the supplied dataset."""

    def __init__(self, data, **kwargs):
        # Private properties
        self._dataset = ''         # Identifier for dataset
        self._SNR_Phi_true = None  # SNR: min(svd(Y_true))/max(svd(E))
        self._SNR_Phi_gauss = None # SNR with Gaussian assumption
        self._SNR_L = None         # SNR: true expression to variance
        self._SNR_phi_true = None  # Min SNR per variable (true)
        self._SNR_phi_gauss = None # Min SNR per variable (Gaussian)

        # Default tolerance from DataModel (assumed)
        self.tol = kwargs.get('tol', np.finfo(float).eps)

        # Analyze the data
        self.analyse_data(data)

    def analyse_data(self, data, **kwargs):
        """Perform analysis on the provided dataset."""
        self._dataset = self.identifier(data)
        self._SNR_Phi_true = self.calc_SNR_Phi_true(data)
        self._SNR_Phi_gauss = self.calc_SNR_Phi_gauss(data)
        self._SNR_L = self.calc_SNR_L(data)
        self._SNR_phi_true = np.min(self.calc_SNR_phi_true(data))
        self._SNR_phi_gauss = np.min(self.calc_SNR_phi_gauss(data))

    # Static methods (class methods in Python)
    @staticmethod
    def alpha():
        """Significance level (default = 0.01)."""
        return 0.01  # Assumed default from DataModel

    @staticmethod
    def type():
        """Network type: 'directed' or 'undirected'."""
        return 'directed'  # Assumed default from DataModel

    @staticmethod
    def tol():
        """Tolerance value for computations."""
        return np.finfo(float).eps  # Assumed default from DataModel

    @staticmethod
    def identifier(data):
        """Get dataset identifier."""
        return data.dataset if isinstance(data, Dataset) else ''

    @staticmethod
    def calc_SNR_Phi_true(data):
        """Calculate SNR_Phi_true: min(svd(true_response))/max(svd(E))."""
        s_true = linalg.svd(data.true_response(), compute_uv=False)
        s_E = linalg.svd(data.E, compute_uv=False) if data.E is not None else np.array([1.0])  # Default if None
        return min(s_true) / max(s_E) if s_E.size > 0 else float('inf')

    @staticmethod
    def calc_SNR_phi_true(data):
        """Calculate SNR_phi_true for each variable."""
        X = data.true_response()
        snr = np.zeros(data.N)
        for i in range(data.N):
            snr[i] = linalg.norm(X[i, :]) / linalg.norm(data.E[i, :]) if linalg.norm(data.E[i, :]) > 0 else float('inf')
        return snr

    @staticmethod
    def calc_SNR_Phi_gauss(data):
        """Calculate SNR_Phi_gauss with Gaussian assumption."""
        alpha = Data.alpha()
        sigma = min(linalg.svd(data.response(), compute_uv=False))
        return sigma / np.sqrt(chi2.ppf(1 - alpha, data.P.size) * data.lambda_[0])

    def calc_SNR_L(self, data):
        alpha = self.alpha()
        try:
            sigma = np.min(linalg.svd(data.true_response(), compute_uv=False))
            denom = np.sqrt(chi2.ppf(1 - alpha, data.P.size) * (data.lambda_ if data.lambda_ is not None else 1.0))
            self._SNR_L = sigma / denom if denom != 0 else float('inf')
        except Exception as e:
            print(f"Error in calc_SNR_L: {e}")
            self._SNR_L = float('nan')
        return self._SNR_L

    @staticmethod
    def calc_SNR_phi_gauss(data):
        """Calculate SNR_phi_gauss for each variable."""
        alpha = Data.alpha()
        Y = data.response()
        SNR = np.zeros(data.N)
        for i in range(data.N):
            SNR[i] = linalg.norm(Y[i, :]) / np.sqrt(chi2.ppf(1 - alpha, data.M) * data.lambda_[0])
        return SNR

    @staticmethod
    def scale_lambda_SNR_L(data, SNR_L):
        """Scale noise variance to achieve desired SNR_L."""
        alpha = Data.alpha()
        s = min(linalg.svd(data.true_response(), compute_uv=False))
        lambda_ = s**2 / (chi2.ppf(1 - alpha, data.P.size) * SNR_L**2)
        return lambda_

    @staticmethod
    def scale_lambda_SNR_E(data, SNR_t):
        """Scale noise variance based on true noise and expression SVD."""
        alpha = Data.alpha()
        s = min(linalg.svd(data.true_response(), compute_uv=False))
        e = max(linalg.svd(data.E, compute_uv=False))
        e2 = s / SNR_t
        lambda_ = np.var(data.E.flatten() * e2 / e)
        return lambda_

    @staticmethod
    def scale_lambda_SNRv(data, SNRv):
        """Scale noise variance to achieve desired SNRv per variable."""
        alpha = Data.alpha()
        Y = data.response()
        lambda_ = np.zeros(data.N)
        for i in range(data.N):
            lambda_[i] = linalg.norm(Y[i, :]) / (chi2.ppf(1 - alpha, data.M) * SNRv**2)
        return lambda_

    @staticmethod
    def irrepresentability(data, net):
        """Calculate the irrepresentable condition for LASSO inference."""
        Y = data.response(net)
        Phi = Y.T
        SIC = np.zeros((net.N, net.N))
        
        for i in range(net.N):
            Phiz = Phi[:, net.A[i, :] == 0]    # Columns where A(i,:) is zero
            Phipc = Phi[:, net.A[i, :] != 0]   # Columns where A(i,:) is nonzero
            if Phipc.size > 0:
                sic = np.abs(Phiz.T @ Phipc @ linalg.pinv(Phipc.T @ Phipc) @ np.sign(net.A[i, net.A[i, :] != 0]).T)
                k = 0
                for j in range(net.N):
                    if net.A[i, j] == 0:
                        SIC[i, j] = sic[k] if k < sic.size else 0
                        k += 1
        
        irr = np.min(1 - SIC[~net.logical()])
        return irr, SIC

    # Property getters
    @property
    def dataset(self):
        return self._dataset

    @property
    def SNR_Phi_true(self):
        return self._SNR_Phi_true

    @property
    def SNR_Phi_gauss(self):
        return self._SNR_Phi_gauss

    @property
    def SNR_L(self):
        if self._SNR_L is None and self.dataset is not None:
            self._SNR_L = self.calc_SNR_L(self.dataset)
        return self._SNR_L

    @property
    def SNR_phi_true(self):
        return self._SNR_phi_true

    @property
    def SNR_phi_gauss(self):
        return self._SNR_phi_gauss
