import numpy as np

class DataModel:
    """Calculates data properties for the supplied dataset."""

    def __init__(self, data, **kwargs):
        super().__init__()
        # Private properties
        self._dataset = ''         # Identifier for dataset
        self._SNR_Phi_true = None  # SNR: min(svd(Y_true))/max(svd(E))
        self._SNR_Phi_gauss = None # SNR with Gaussian assumption
        self._SNR_L = None         # SNR: true expression to variance
        self._SNR_phi_true = None  # Min SNR per variable (true)
        self._SNR_phi_gauss = None # Min SNR per variable (Gaussian)

        # Set tolerance from kwargs or use inherited default
        self._tol = kwargs.get('tol', self.tol())

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

    # Static methods inherited from DataModel are overridden to use class methods
    @classmethod
    def identifier(cls, data):
        """Get dataset identifier."""
        return data.dataset if isinstance(data, Dataset) else ''

    @classmethod
    def calc_SNR_Phi_true(cls, data):
        """Calculate SNR_Phi_true: min(svd(true_response))/max(svd(E))."""
        s_true = linalg.svd(data.true_response(), compute_uv=False)
        s_E = linalg.svd(data.E, compute_uv=False)
        return min(s_true) / max(s_E) if s_E.size > 0 else float('inf')

    @classmethod
    def calc_SNR_phi_true(cls, data):
        """Calculate SNR_phi_true for each variable."""
        X = data.true_response()
        snr = np.zeros(data.N)
        for i in range(data.N):
            snr[i] = linalg.norm(X[i, :]) / linalg.norm(data.E[i, :]) if linalg.norm(data.E[i, :]) > 0 else float('inf')
        return snr

    @classmethod
    def calc_SNR_Phi_gauss(cls, data):
        """Calculate SNR_Phi_gauss with Gaussian assumption."""
        alpha = cls.alpha()
        sigma = min(linalg.svd(data.response(), compute_uv=False))
        return sigma / np.sqrt(chi2.ppf(1 - alpha, data.P.size) * data.lambda_[0])

    @classmethod
    def calc_SNR_L(cls, data):
        """Calculate SNR_L: true expression to variance relationship."""
        alpha = cls.alpha()
        sigma = min(linalg.svd(data.true_response(), compute_uv=False))
        return sigma / np.sqrt(chi2.ppf(1 - alpha, data.P.size) * data.lambda_[0])

    @classmethod
    def calc_SNR_phi_gauss(cls, data):
        """Calculate SNR_phi_gauss for each variable."""
        alpha = cls.alpha()
        Y = data.response()
        SNR = np.zeros(data.N)
        for i in range(data.N):
            SNR[i] = linalg.norm(Y[i, :]) / np.sqrt(chi2.ppf(1 - alpha, data.M) * data.lambda_[0])
        return SNR

    @classmethod
    def scale_lambda_SNR_L(cls, data, SNR_L):
        """Scale noise variance to achieve desired SNR_L."""
        alpha = cls.alpha()
        s = min(linalg.svd(data.true_response(), compute_uv=False))
        lambda_ = s**2 / (chi2.ppf(1 - alpha, data.P.size) * SNR_L**2)
        return lambda_

    @classmethod
    def scale_lambda_SNR_E(cls, data, SNR_t):
        """Scale noise variance based on true noise and expression SVD."""
        alpha = cls.alpha()
        s = min(linalg.svd(data.true_response(), compute_uv=False))
        e = max(linalg.svd(data.E, compute_uv=False))
        e2 = s / SNR_t
        lambda_ = np.var(data.E.flatten() * e2 / e)
        return lambda_

    @classmethod
    def scale_lambda_SNRv(cls, data, SNRv):
        """Scale noise variance to achieve desired SNRv per variable."""
        alpha = cls.alpha()
        Y = data.response()
        lambda_ = np.zeros(data.N)
        for i in range(data.N):
            lambda_[i] = linalg.norm(Y[i, :]) / (chi2.ppf(1 - alpha, data.M) * SNRv**2)
        return lambda_

    @classmethod
    def irrepresentability(cls, data, net):
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
        return self._SNR_L

    @property
    def SNR_phi_true(self):
        return self._SNR_phi_true

    @property
    def SNR_phi_gauss(self):
        return self._SNR_phi_gauss

    @property
    def tol(self):
        return self._tol
