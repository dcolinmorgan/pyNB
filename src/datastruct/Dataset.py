import numpy as np
from numpy import linalg
import os
from datetime import datetime
from datastruct.Exchange import Exchange
from datastruct.Network import Network
from datastruct.Experiment import Experiment

class Dataset(Exchange):
    """The Dataset class stores data complementary to a datastruct.Network."""

    def __init__(self, *args):
        super().__init__()

        # Default attribute initialization
        self.dataset = ''  # Name of the dataset
        self.network = ''  # Name of complementary network
        self.P = None      # Observed/assumed perturbations
        self.F = None      # Perturbation noise
        self.cvP = None    # Covariance of P
        self.sdP = None    # Measurement point variation of P
        self.Y = None      # Observed expression response
        self.E = None      # Expression noise
        self.cvY = None    # Covariance of noisy Y
        self.sdY = None    # Measurement point variation of Y
        self.lambda_ = None # Noise variance (renamed to avoid Python keyword conflict)
        # self.SNR_L = None   # Signal-to-noise ratio
        # self.SNR_Wiki = None # Wikipedia SNR (added by Deniz)
        self.A = None       # Network
        self.G = None       # Static gain model
        self.names = []
        self.description = ''
        for arg in args:
            if isinstance(arg, (Network, Experiment)):
                self.populate(arg)
        
        # Hidden properties
        self._N = None      # Number of variables in A
        self._M = None      # Number of experiments
        self.created = {
            'creator': os.getlogin(),
            'time': datetime.now(),
            'id': '',
            'nexp': ''
        }
        self.tol = np.finfo(float).eps  # Machine epsilon

        # Process input arguments
        for arg in args:
            if hasattr(arg, 'A'):  # Assuming Network-like object
                self.populate(arg)
            elif hasattr(arg, 'Y'):  # Assuming Experiment or Dataset-like object
                experiment = vars(arg)
                experiment['Y'] = arg.trueY() if hasattr(arg, 'trueY') else arg.Y
                self.populate(experiment)
                self.created['id'] = str(round(np.linalg.cond(self.Y - self.E) * 10000))
            elif isinstance(arg, dict):  # Struct-like input
                self.populate(arg)
                self.created['id'] = str(round(np.linalg.cond(self.Y - self.E) * 10000))
                
            if isinstance(arg, Experiment):
                experiment = vars(arg)
                experiment['Y'] = arg.trueY()
                self.populate(experiment)
                # self.created['id'] = str(round(np.linalg.cond(self.Y - self.E) * 10000))
                
            if isinstance(arg, Network):
                self.populate(arg)
        
        self.setname()

    # Getter for M (number of experiments)
    @property
    def M(self):
        return self.P.shape[1] if self.P is not None else 0

    # Getter for N (number of variables)
    @property
    def N(self):
        return self.P.shape[0] if self.P is not None else 0

    # Setter for lambda
    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, lambda_val):
        if not isinstance(lambda_val, np.ndarray):
            lambda_val = np.array(lambda_val)
        if lambda_val.ndim == 0:  # Scalar
            lambda_val = np.array([lambda_val, 0])
        elif lambda_val.size == self.N:
            lambda_val = np.concatenate([lambda_val, np.zeros_like(lambda_val)])
        if len(lambda_val) % 2 != 0:
            raise ValueError("Something is wrong with the size of lambda. Help!")
        self._lambda = lambda_val

    # Signal-to-noise ratio (SNR_L)
    @property
    def SNR_L(self):
        alpha = 0.01
        sigma = np.min(linalg.svd(self.true_response()))
        from scipy.stats import chi2
        return sigma / np.sqrt(chi2.ppf(1 - alpha, self.P.size) * self.lambda_[0])

    # Wikipedia SNR
    @property
    def SNR_Wiki(self):
        P = self.P
        reps = np.sum(P != 0, axis=1)
        cs = np.cumsum(reps)
        sd = np.zeros(P.shape[0])
        m = np.zeros(P.shape[0])
        y = self.Y[:, :cs[0]]
        sd[0] = np.abs(np.std(y))
        m[0] = np.abs(np.mean(y))
        for j in range(P.shape[0] - 1):
            y = self.Y[:, cs[j]:cs[j + 1]]
            sd[j + 1] = np.abs(np.std(y))
            m[j + 1] = np.abs(np.mean(y))
        return np.median(m) / np.median(sd)

    # Helper methods
    def Phi(self):
        return self.Y.T

    def Xi(self):
        return -self.P.T

    def Upsilon(self):
        return self.E.T

    def Pi(self):
        return -self.F.T

    def Psi(self):
        return np.hstack([self.Phi(), self.Xi()])

    def Omicron(self):
        return np.hstack([self.Upsilon(), self.Pi()])

    def setname(self, namestruct=None):
        if namestruct is None:
            namestruct = self.created
        if not isinstance(namestruct, dict):
            raise ValueError("Input argument must be name/value pairs in struct form")
        
        namer = self.created.copy()
        for key, value in namestruct.items():
            if key in namer:
                namer[key] = value

        SNR_L = '0' if self.lambda_ is None else str(round(self.SNR_L * 1000))
        SNR_Wiki = '0' if self.lambda_ is None else str(self.SNR_Wiki)
        network_id = self.network.split('-ID')[-1] if '-ID' in self.network else ''
        self.dataset = (f"{namer['creator']}-ID{network_id}-D{namer['time'].strftime('%Y%m%d')}"
                        f"-N{self.P.shape[0] if self.P is not None else 0}"
                        f"-E{self.P.shape[1] if self.P is not None else 0}"
                        f"-SNR{SNR_L}-SNR_Wiki{SNR_Wiki}-IDY{namer['id']}")

    @property
    def names(self):
        if not self._names:
            return [f"G{i:0{int(np.log10(self.N)) + 1}d}" for i in range(1, self.N + 1)]
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    def std(self):
        if self.lambda_.size == 2:
            sdY = np.sqrt(self.lambda_[0]) * np.ones_like(self.P)
            sdP = np.sqrt(self.lambda_[1]) * np.ones_like(self.P)
        else:
            sdY = np.sqrt(self.lambda_[:self.N]).reshape(-1, 1) * np.ones((1, self.P.shape[1]))
            sdP = np.sqrt(self.lambda_[self.N:]).reshape(-1, 1) * np.ones((1, self.P.shape[1]))
        return sdY, sdP

    def scaleSNR(self, net, SNR):
        newdata = Dataset(self, net)
        sY = linalg.svd(self.true_response())
        sE = linalg.svd(self.E)
        scale = (1 / SNR) * np.min(sY) / np.max(sE)
        newdata.lambda_ = scale**2 * self.lambda_
        newdata.E = scale * self.E
        return newdata

    def std_normalize(self, dim=2):
        newdata = Dataset(self)
        Y = self.response()
        mu = np.mean(Y, axis=dim-1, keepdims=True)
        sigma = np.std(Y, axis=dim-1, ddof=0, keepdims=True)  # Population std
        Yhat = (Y - mu) / sigma
        newdata.Y = Yhat
        newdata.E = np.zeros_like(Yhat)
        newdata.cvP = None
        newdata.cvY = None
        return newdata

    def populate(self, input):
        if isinstance(input, Network):
            self.network = input
            self.Y = input.A @ input.P if input.P is not None else input.A
            self.P = input.P if input.P is not None else np.eye(input.A.shape[0])
        elif isinstance(input, Experiment):
            self.network = Network(input._G)  # Create a Network from Experiment's G
            self.Y = input.noiseY()
            self.P = input._P
            self.lambda_ = np.var(input._E) if input._E is not None else 1.0
        self.X = self.Y

    def response(self, net=None):
        if net and hasattr(net, 'G'):
            X = net.G @ self.P
            return X + self.E
        return self.Y

    def true_response(self):
        if  self.network is None or self.network.G is None:
            if self.network is None and self.Y is not None:
                self.network = Network(np.eye(self.Y.shape[0]))
            self.network.G = self.network.A if self.network.G is None else self.network.G
        return self.network.G # if isinstance(self.network.G, np.ndarray) else self.network.G.A
