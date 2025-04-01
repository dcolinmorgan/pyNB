import numpy as np
from scipy import linalg
from scipy.stats import chi2
from datastruct.Network import Network

class Experiment:
    """Class to simulate perturbation experiments on linear systems."""

    def __init__(self, *args):
        # Protected properties
        self._A = None          # True network model
        self._G = None          # True static gain matrix
        self._P = np.array([])  # Designed perturbations
        self._Yinit = np.array([])  # Initial observed responses
        self._E = np.array([])  # Expression noise
        self._F = np.array([])  # Perturbation noise

        # Public properties
        self.lambda_ = np.array([1, 0])  # Variance of Gaussian noise
        self.alpha = 0.05                # Significance level
        self.nExp = float('inf')         # Number of experiments
        self.mag = 1.0                   # Perturbation magnitude multiplier
        self.maxmag = 2.0                # Maximum perturbation magnitude
        self.SignalThreshold = 1.0       # Signal threshold
        self.tol = np.finfo(float).eps   # Tolerance for calculations
        self.nnzP = None                 # Number of non-zero elements in perturbations
        self.SNR = None                  # Signal-to-noise ratio

        # Hidden properties
        self._N = None                   # Number of variables in A
        self._M = None                   # Number of samples

        # Handle input arguments
        if not args:
            print("Warning: Make sure to supply a Network before further experimentation!")
        if len(args) >= 1:
            self.A = args[0]  # Set A and initialize related properties
        if len(args) >= 2:
            if not isinstance(args[1], np.ndarray):
                raise ValueError("Perturbation must be a NumPy array")
            self._P = args[1]
        if len(args) >= 3:
            if not isinstance(args[2], np.ndarray):
                raise ValueError("Initial response must be a NumPy array")
            self._Yinit = args[2]

    # Getter and Setter for A
    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, net):
        if self._A is None:
            if isinstance(net, np.ndarray):
                self._A = net
            elif hasattr(net, 'A'):  # Assuming Network-like object
                self._A = net.A
            else:
                raise ValueError("Network must be a NumPy array or Network object")
            self._G = -linalg.pinv(self._A)  # Pseudo-inverse
            self._P = np.vstack([1, np.zeros((self.N - 1, 1))])
            if self.nnzP is None:
                self.nnzP = self.N
        else:
            print("Warning: True A already set, will not change it!")

    # Getter for N
    @property
    def N(self):
        return self._A.shape[0] if self._A is not None else 0

    # Getter for M
    @property
    def M(self):
        return self._P.shape[1] if self._P.size > 0 else 0

    # Setter for alpha
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if not 0 <= value <= 1:
            raise ValueError("Significance level must be in the range [0,1]")
        self._alpha = value

    # Setter for lambda
    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, lambda_val):
        lambda_val = np.atleast_1d(lambda_val)
        if lambda_val.size == 1:
            lambda_val = np.array([lambda_val[0], 0])
        elif lambda_val.size == self._P.shape[0]:
            lambda_val = np.concatenate([lambda_val, np.zeros_like(lambda_val)])
        if len(lambda_val) % 2 != 0:
            raise ValueError("Something is wrong with the size of lambda. Help!")
        self._lambda = lambda_val

    # Setter for SignalThreshold
    @property
    def SignalThreshold(self):
        return self._SignalThreshold

    @SignalThreshold.setter
    def SignalThreshold(self, value):
        if value <= 0:
            raise ValueError("SignalThreshold must be > 0")
        self._SignalThreshold = value

    # Setter for nExp
    @property
    def nExp(self):
        return self._nExp

    @nExp.setter
    def nExp(self, value):
        if not isinstance(value, (int, float)) or value < 1:
            raise ValueError("# experiments must be a positive number")
        self._nExp = value if value != float('inf') else float('inf')

    def Pinit(self, P):
        """Set an initial perturbation, overwriting current perturbation."""
        self._P = P

    def set_Yinit(self, Yinit):
        """Set initial response."""
        self._Yinit = Yinit

    def terminate(self, condition=None):
        """Check if stop conditions are met."""
        if self._P.shape[1] >= self.nExp:
            return True
        if condition == 'ST':  # Signal Threshold
            s = linalg.svd(self.noiseY())[1]  # Singular values
            return min(s) > self.SignalThreshold and self._P.shape[1] >= self.N
        elif condition == 'SCT':  # Scaled Threshold
            k = self._P.shape[1]
            s = linalg.svd(self.noiseY() / np.sqrt(chi2.ppf(1 - self.alpha, self.N * k) * self.var()))[1]
            return min(s) > self.SignalThreshold and self._P.shape[1] >= self.N
        return False
    
    def populate(self, input):
        if isinstance(input, Network):
            if input.A is None:
                raise ValueError("Network A matrix must be set")
            n = input.A.shape[0]
            self._G = input.G if input.G is not None else np.eye(n)
            self._P = input.P if input.P is not None else np.eye(n)
            self._S = input.S
        else:
            raise ValueError("Input must be a Network object")
        
    def signal(self):
        if self._G is None or self._P is None:
            raise ValueError("G and P must be set before computing signal")
        return self._G @ self._P

    def noise(self):
        if self._E is None:
            raise ValueError("Noise (E) must be set before accessing")
        return self._E

    # def trueY(self):
    #     """Generate expression without noise for the complete dataset."""
    #     pre = self._Yinit.shape[1] if self._Yinit.size > 0 else 0
    #     if pre >= self._P.shape[1]:
    #         return self._Yinit
    #     return np.hstack([self._Yinit, self._G @ self._P[:, pre:]])
    def trueY(self):
        if self._Y is None:
            self.gaussian()
        pre = 0  # Adjust based on your intent (e.g., pre = self._P.shape[1] // 2)
        signal_part = self._G @ self._P[:, pre:]
        return np.hstack([self._Y[:, :pre] if pre > 0 else np.zeros((self._Y.shape[0], 0)), signal_part])

    # def noiseY(self):
    #     if not hasattr(self,'_Y'):
    #         self.gaussian()
    #     while self._E.shape[1] < self._P.shape[1]:
    #         self._E = np.hstack((self._E, np.random.randn(self._E.shape[0], self._P.shape[1] - self._E.shape[1])))
    #     return self._Y
    
    def noiseP(self):
        """Generate perturbations with noise for the complete dataset."""
        while self._F.shape[1] < self._P.shape[1]:
            self.gaussian()
        return self._P - self._F[:, :self._P.shape[1]]

    def var(self):
        """Calculate point-wise variance of Y."""
        if self.lambda_.size == 2:
            return self.lambda_[0] * np.ones_like(self._P)
        return self.lambda_[:self.N].reshape(-1, 1) * np.ones((1, self._P.shape[1]))

    def sparse(self, nnzP=None):
        """Make each perturbation sufficiently sparse naively."""
        nnzP = nnzP if nnzP is not None else self.nnzP
        p = self._P[:, -1].copy()
        nZero = self.N - nnzP
        sortIndex = np.argsort(np.abs(p))
        minIndex = sortIndex[:nZero]
        p[minIndex] = 0
        self._P[:, -1] = p

    def initE(self, E):
        """Set expression noise."""
        self._E = E

    def initF(self, F):
        """Set perturbation noise."""
        self._F = F

    # def gaussian(self):
    #     """Generate Gaussian noise with variance lambda."""
    #     if self.lambda_.size == 1:
    #         self._E = np.hstack([self._E, np.sqrt(self.lambda_[0]) * np.random.randn(self.N, 1)])
    #         self._F = np.hstack([self._F, np.zeros((self.N, 1))])
    #     elif self.lambda_.size == 2:
    #         self._E = np.hstack([self._E, np.sqrt(self.lambda_[0]) * np.random.randn(self.N, 1)])
    #         self._F = np.hstack([self._F, np.sqrt(self.lambda_[1]) * np.random.randn(self.N, 1)])
    #     elif self.lambda_.size == self.N:
    #         self._E = np.hstack([self._E, np.sqrt(self.lambda_) * np.random.randn(self.N, 1)])
    #         self._F = np.hstack([self._F, np.zeros((self.N, 1))])
    #     elif self.lambda_.size == 2 * self.N:
    #         self._E = np.hstack([self._E, np.sqrt(self.lambda_[:self.N]) * np.random.randn(self.N, 1)])
    #         self._F = np.hstack([self._F, np.sqrt(self.lambda_[self.N:]) * np.random.randn(self.N, 1)])

    def gaussian(self, scale=1):
        if self._P is None:
            raise ValueError("P must be set before generating noise")
        if self._G is None:
            self._G = np.eye(self._P.shape[0])
        n_rows, n_cols = self._G.shape[0], self._P.shape[1]
        self._E = scale * np.random.randn(n_rows, n_cols)
        self._Y = self.signal() + self.noise()
        return self

    def noiseY(self):
        if not hasattr(self,'_Y'):# is None:
            print("Y is None, calling gaussian")
            self.gaussian()
        else:
            print("Y already set")
        print(f"E shape: {self._E.shape}, P shape: {self._P.shape}")
        if len(self._E.shape) < 2 or self._E.shape[1] < self._P.shape[1]:
            n_rows = self._G.shape[0]
            n_cols_needed = self._P.shape[1]
            if len(self._E.shape) < 2:
                self._E = np.zeros((n_rows, 0))
            extra_cols = np.random.randn(n_rows, n_cols_needed - self._E.shape[1])
            self._E = np.hstack((self._E, extra_cols))
            print(f"Adjusted E shape: {self._E.shape}")
        return self._Y

    @property
    def SNR(self):
        """Get signal-to-noise ratio."""
        if self._SNR is None:
            sY = linalg.svd(self.trueY())[1]
            sE = linalg.svd(self._E)[1]
            return min(sY) / max(sE) if sE.size > 0 else float('inf')
        return self._SNR

    @SNR.setter
    def SNR(self, value):
        self._SNR = value

    def scaleSNR(self, SNR):
        """Scale noise variance to achieve desired SNR."""
        self.noiseY()
        sY = linalg.svd(self.trueY())[1]
        sE = linalg.svd(self._E)[1]
        scale = (1 / SNR) * min(sY) / max(sE)
        self.lambda_ = scale**2 * self.lambda_
        self._E = scale * self._E

    def scaleSNRm(self, SNRm, nvar=None):
        """Scale noise variance to achieve desired SNRm."""
        lambdaOld = self.lambda_.copy()
        nvar = nvar if nvar is not None else self.N * self.M
        sY = linalg.svd(self.trueY())[1]
        lambda_new = min(sY)**2 / (chi2.ppf(1 - self.alpha, nvar) * SNRm**2)
        self.lambda_ = np.array([lambda_new, 0]) if self.lambda_.size == 2 else lambda_new * np.ones_like(self.lambda_)
        self._E = np.sqrt(lambda_new / lambdaOld[0]) * self._E

    def signify(self):
        """Scale perturbations to their sign times magnitude."""
        self._P = self.mag * np.sign(self._P)

    def SVDE(self):
        """SVD-based perturbation design."""
        k = self._P.shape[1]
        if k + 1 <= self.N:
            newdir = self._gram_schmidt_orth(k + 1)
            self._P = np.hstack([self._P, self.mag * newdir[:, [k]]])
        if k > 0:
            U, S, Vh = linalg.svd(self.noiseY())
            s = np.diag(S) if S.ndim == 1 else S
            if min(s) < self.SignalThreshold:
                self._P = np.hstack([self._P, np.zeros((self.N, 1))])
                for j in range(min(s.size, self.N)):
                    if s[j] > self.SignalThreshold:
                        self._P[:, -1] += (self.SignalThreshold / s[j]) * (self._P[:, :k] @ Vh[j, :].T)

    def _gram_schmidt_orth(self, k):
        """Simple Gram-Schmidt orthogonalization (placeholder)."""
        P = self._P[:, :k-1] if k > 1 else np.zeros((self.N, 0))
        new_vec = np.random.randn(self.N, 1)
        for i in range(P.shape[1]):
            new_vec -= (new_vec.T @ P[:, i]) / (P[:, i].T @ P[:, i]) * P[:, i, np.newaxis]
        return new_vec / linalg.norm(new_vec)

    # Additional methods like BCSE, BSVE, etc., can be added similarly...
