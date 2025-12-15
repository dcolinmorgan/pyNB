import numpy as np
from numpy import linalg
from datastruct.Network import Network
from .Model import Model
from typing import Union, List, Optional
from sklearn.metrics import roc_auc_score


class CompareModels:
    """Compares multiple network models against a reference."""
    
    def __init__(self, ref_net: Network, net_list: Union[np.ndarray, Network, List[Network]]) -> None:
        """Initialize comparison between networks.
        
        Args:
            ref_net: Reference Network object to compare against
            net_list: Either a single Network object or a numpy array of networks
                     If numpy array, should be of shape (N, N, M) where M is number of networks
        """
        self._ref = Model(ref_net)
        self._models: List[Model] = []
        
        if isinstance(net_list, Network):
            if net_list.A is not None and net_list.A.ndim == 3:
                # Network object containing 3D array (e.g. Lasso path)
                self._models = [Model(Network(net_list.A[:, :, i])) for i in range(net_list.A.shape[2])]
            else:
                # Single network comparison
                self._models = [Model(net_list)]
        elif isinstance(net_list, np.ndarray):
            # Multiple network comparison (numpy array)
            if net_list.ndim == 3:
                self._models = [Model(Network(net_list[:, :, i])) for i in range(net_list.shape[2])]
            else:
                 # Assume list of networks? Or 2D array?
                 # Original code: net_list.transpose(2, 0, 1) implies 3D array (N, N, M)
                 # But transpose(2, 0, 1) on (N, N, M) -> (M, N, N)
                 # Then iterating gives (N, N) matrices.
                 # My fix above for Network object does manual slicing.
                 
                 # Let's keep original logic for numpy array but be safer
                 # If it's 2D, treat as single network?
                 if net_list.ndim == 2:
                     self._models = [Model(Network(net_list))]
                 else:
                     # Fallback or error?
                     pass
        elif isinstance(net_list, list):
             self._models = [Model(net) if isinstance(net, Network) else Model(Network(net)) for net in net_list]
            
        self._analyze()

    def _analyze(self) -> None:
        """Compute comparison metrics."""
        ref_A = self._ref.data.A
        if ref_A is None:
            return
            
        n_nodes = ref_A.shape[0]
        n_models = len(self._models)
        
        # Initialize metric arrays
        self._afronorm = np.zeros(n_models)
        self._F1 = np.zeros(n_models)
        self._nlinks = np.zeros(n_models)
        self._TP = np.zeros(n_models)
        self._TN = np.zeros(n_models)
        self._FP = np.zeros(n_models)
        self._FN = np.zeros(n_models)
        self._sen = np.zeros(n_models)
        self._spe = np.zeros(n_models)
        self._comspe = np.zeros(n_models)
        self._pre = np.zeros(n_models)
        self._TPTN = np.zeros(n_models)
        self._structsim = np.zeros(n_models)
        self._MCC = np.zeros(n_models)
        self._FEL = np.zeros(n_models)
        self._AUROC = np.zeros(n_models)
        
        # Get reference network binary matrix
        ref_binary = ref_A != 0
        
        for i, model in enumerate(self._models):
            pred_A = model.data.A
            if pred_A is None:
                continue
            
            # Basic metrics
            self._afronorm[i] = linalg.norm(ref_A - pred_A, 'fro')
            
            # Get predicted network binary matrix
            pred_binary = pred_A != 0
            
            # Calculate TP, TN, FP, FN using MATLAB-style formulation
            self._TP[i] = np.sum(np.logical_and(ref_binary, pred_binary))
            self._TN[i] = np.sum(np.logical_and(~ref_binary, ~pred_binary))
            self._FP[i] = np.sum(np.logical_and(~ref_binary, pred_binary))
            self._FN[i] = np.sum(np.logical_and(ref_binary, ~pred_binary))
            
            # Derived metrics
            self._nlinks[i] = np.sum(pred_binary)
            self._TPTN[i] = self._TP[i] + self._TN[i]
            self._structsim[i] = self._TPTN[i] / (n_nodes * n_nodes)
            
            # Rate metrics
            self._sen[i] = self._TP[i] / (self._TP[i] + self._FN[i]) if (self._TP[i] + self._FN[i]) > 0 else 0
            self._spe[i] = self._TN[i] / (self._TN[i] + self._FP[i]) if (self._TN[i] + self._FP[i]) > 0 else 0
            self._comspe[i] = 1 - self._spe[i]
            self._pre[i] = self._TP[i] / (self._TP[i] + self._FP[i]) if (self._TP[i] + self._FP[i]) > 0 else 0
            
            # F1 score
            self._F1[i] = 2 * (self._pre[i] * self._sen[i]) / (self._pre[i] + self._sen[i]) if (self._pre[i] + self._sen[i]) > 0 else 0
            
            # MCC (Matthews Correlation Coefficient)
            numerator = self._TP[i] * self._TN[i] - self._FP[i] * self._FN[i]
            denominator = np.sqrt((self._TP[i] + self._FP[i]) * (self._TP[i] + self._FN[i]) * 
                                (self._TN[i] + self._FP[i]) * (self._TN[i] + self._FN[i]))
            self._MCC[i] = numerator / denominator if denominator > 0 else 0
            
            # FEL (Fraction of Existing Links)
            self._FEL[i] = self._FP[i] / np.sum(ref_binary) if np.sum(ref_binary) > 0 else 0

            # AUROC
            # roc_auc_score requires binary labels and probability scores.
            # If pred_binary is binary, it works but might be trivial.
            # If pred_A has weights, we should use them?
            # Original code used pred_binary.
            try:
                self._AUROC[i] = roc_auc_score(ref_binary.flatten(), pred_binary.flatten())
            except ValueError:
                # Handle case where only one class is present in y_true
                self._AUROC[i] = 0.5

    @property
    def afronorm(self) -> np.ndarray:
        """Get array of Frobenius norms between reference and each model."""
        return self._afronorm

    @property
    def F1(self) -> np.ndarray:
        """Get array of F1 scores between reference and each model."""
        return self._F1
        
    @property
    def nlinks(self) -> np.ndarray:
        """Get array of number of links in each estimated network."""
        return self._nlinks
        
    @property
    def TP(self) -> np.ndarray:
        """Get array of true positive counts for each model."""
        return self._TP
        
    @property
    def TN(self) -> np.ndarray:
        """Get array of true negative counts for each model."""
        return self._TN
        
    @property
    def FP(self) -> np.ndarray:
        """Get array of false positive counts for each model."""
        return self._FP
        
    @property
    def FN(self) -> np.ndarray:
        """Get array of false negative counts for each model."""
        return self._FN
        
    @property
    def sen(self) -> np.ndarray:
        """Get array of sensitivity (TP/(TP+FN)) for each model."""
        return self._sen
        
    @property
    def spe(self) -> np.ndarray:
        """Get array of specificity (TN/(TN+FP)) for each model."""
        return self._spe
        
    @property
    def comspe(self) -> np.ndarray:
        """Get array of complementary specificity (1-Specificity) for each model."""
        return self._comspe
        
    @property
    def pre(self) -> np.ndarray:
        """Get array of precision (TP/(TP+FP)) for each model."""
        return self._pre
        
    @property
    def TPTN(self) -> np.ndarray:
        """Get array of TP+TN counts for each model."""
        return self._TPTN
        
    @property
    def structsim(self) -> np.ndarray:
        """Get array of structural similarity ((TP+TN)/N^2) for each model."""
        return self._structsim
        
    @property
    def MCC(self) -> np.ndarray:
        """Get array of Matthews correlation coefficients for each model."""
        return self._MCC
        
    @property
    def FEL(self) -> np.ndarray:
        """Get array of fraction of existing links (FP/TP_ref) for each model."""
        return self._FEL

    @property
    def AUROC(self) -> np.ndarray:
        """Get array of AUROC for each model."""
        return self._AUROC
