import numpy as np
from scipy import linalg
from datastruct.Network import Network
import os
import pandas as pd
import matplotlib.pyplot as plt

class CompareModels:
    """Calculates difference measures between weighted network adjacency matrices."""

    def __init__(self, *args):
        # Hidden properties
        self.tol = np.finfo(float).eps  # Link strength tolerance

        # Public properties
        self._A = None          # Gold standard network
        self.zetavec = None     # Inference parameter (not used in code)

        # Hidden private properties
        self._SA = None         # Singular values of true A
        self._UA = None         # Left singular vectors of true A
        self._VA = None         # Right singular vectors of true A
        self._LA = None         # Eigenvalues of true A
        self._QA = None         # Eigenvectors of true A
        self._DGA = None        # Adjacency matrix of A (logical)
        self._STA = None        # Signed topology of A
        self._N = None          # Number of nodes
        self._ntl = None        # Number of true links
        self._npl = None        # Number of possible links

        # Measure properties (initialized as empty lists)
        self.abs2norm = []
        self.rel2norm = []
        self.maee = []
        self.mree = []
        self.mase = []
        self.mrse = []
        self.masde = []
        self.mrsde = []
        self.maeve = []
        self.mreve = []
        self.maede = []
        self.mrede = []
        self.afronorm = []
        self.rfronorm = []
        self.al1norm = []
        self.rl1norm = []
        self.n0larger = []
        self.r0larger = []
        self.ncs = []
        self.sst = []
        self.sst0 = []
        self.plc = []
        self.nlinks = []
        self.TP = []
        self.TN = []
        self.FP = []
        self.FN = []
        self.sen = []
        self.spe = []
        self.comspe = []
        self.pre = []
        self.TPTN = []
        self.structsim = []
        self.F1 = []
        self.MCC = []
        self.TR = []
        self.TZ = []
        self.FI = []
        self.FR = []
        self.FZ = []
        self.dirsen = []
        self.dirspe = []
        self.dirprec = []
        self.SMCC = []

        # Handle input arguments
        if len(args) >= 1:
            self.A = args[0]
        if len(args) >= 2:
            self.compare(args[1])

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, net):
        if self._A is not None and net is not None:
            print("Warning: True A already set, overwriting")
        if isinstance(net, Network):
            if net.A.ndim > 2:
                raise ValueError("3D matrices are not allowed as golden standard")
            self._A = net.A
        else:
            net = np.asarray(net)
            if net.ndim > 2:
                raise ValueError("3D matrices are not allowed as golden standard")
            self._A = net
        self.setA(self._A)

    def setA(self, net):
        """Calculate network properties."""
        self._A = net
        U, S, Vh = linalg.svd(self._A)
        self._UA = U
        self._SA = S
        self._VA = Vh.T  # Vh is V transpose in scipy

        if self.issquare():
            L, Q = linalg.eig(self._A)
            idx = np.argsort(np.abs(L))[::-1]
            self._LA = L[idx]
            self._QA = Q[:, idx]

        Z = np.zeros_like(self._A)
        self._DGA = np.abs(self._A) > self.tol
        self._STA = Z.copy()
        self._STA[self._A > self.tol] = 1
        self._STA[self._A < -self.tol] = -1
        self._N = self._A.shape[0]
        self._npl = self._A.size
        self._ntl = np.sum(self._DGA)

    def delA(self):
        """Delete the compare network."""
        self.setA(None)

    def sst2z(self):
        """Calculate similarity of signed topology for an empty network."""
        return np.sum(self._A == 0) / self._npl

    def system_measures(self, Alist):
        """Calculate system measures."""
        for i in range(Alist.shape[2]):
            T = Alist[:, :, i]
            self.abs2norm.append(linalg.norm(T - self._A))
            try:
                self.rel2norm.append(linalg.norm(linalg.pinv(T) @ T - self._A))
            except linalg.LinAlgError:
                self.rel2norm.append(np.nan)
            self.maee.append(np.max(np.abs(T - self._A)))
            self.mree.append(np.max(np.abs(T - self._A) / np.maximum(np.max(np.abs(T), axis=1), self.tol)))
            S = linalg.svd(T, compute_uv=False)
            self.mase.append(np.max(np.abs(S - self._SA)))
            self.mrse.append(np.max(np.abs(S - self._SA) / np.maximum(S, self.tol)))
            temp = self._UA.T @ T @ self._VA
            self.masde.append(np.max(np.abs(np.diag(temp) - self._SA)))
            self.mrsde.append(np.max(np.abs(np.diag(temp) - self._SA) / np.maximum(np.diag(temp), self.tol)))
            L = linalg.eigvals(T)
            idx = np.argsort(np.abs(L))[::-1]
            L = L[idx]
            self.maeve.append(np.max(np.abs(L - self._LA)))
            self.mreve.append(np.max(np.abs(L - self._LA) / np.maximum(np.abs(L), self.tol)))
            temp = linalg.pinv(self._QA) @ T @ self._QA
            temp2 = np.abs(np.diag(temp) - self._LA)
            self.maede.append(np.max(temp2))
            self.mrede.append(np.max(temp2 / np.maximum(np.abs(np.diag(temp)), self.tol)))
            self.afronorm.append(linalg.norm(T - self._A, ord='fro'))
            try:
                self.rfronorm.append(linalg.norm(linalg.pinv(T) @ (T - self._A), ord='fro'))
            except linalg.LinAlgError:
                self.rfronorm.append(np.nan)
            self.al1norm.append(np.sum(np.abs(T[~self._DGA])))
            self.rl1norm.append(np.sum(np.abs(T[~self._DGA])) / np.sum(np.abs(self._A)))
            min_nz = np.min(np.abs(self._A[self._DGA])) if np.any(self._DGA) else self.tol
            self.n0larger.append(np.sum(np.abs(T[~self._DGA]) > min_nz))
            self.r0larger.append(self.n0larger[-1] / np.sum(~self._DGA) if np.any(~self._DGA) else 0)

    def topology_measures(self, Alist):
        """Calculate topological measures."""
        for i in range(Alist.shape[2]):
            T = Alist[:, :, i]
            STopoT = np.zeros_like(self._A)
            STopoT[T > self.tol] = 1
            STopoT[T < -self.tol] = -1
            temp = self._STA == STopoT
            self.ncs.append(np.sum(temp))
            self.sst.append(np.sum(temp) / self._npl)
            self.sst0.append(np.sum(temp[self._DGA]) / self._ntl if self._ntl > 0 else 0)

    def correlation_measures(self, Alist):
        """Calculate correlation measures."""
        for i in range(Alist.shape[2]):
            T = Alist[:, :, i]
            self.plc.append(np.corrcoef(self._A.ravel(), T.ravel())[0, 1])

    def graph_measures(self, Alist):
        """Calculate non-signed graph measures."""
        Z = np.zeros_like(self._A)
        for i in range(Alist.shape[2]):
            T = Alist[:, :, i]
            DiGraphT = np.abs(T) > self.tol
            self.nlinks.append(np.sum(DiGraphT))
            self.TP.append(np.sum(self._DGA & DiGraphT))
            self.TN.append(np.sum(~self._DGA & ~DiGraphT))
            self.FP.append(np.sum(~self._DGA & DiGraphT))
            self.FN.append(np.sum(self._DGA & ~DiGraphT))
            self.sen.append(self.TP[-1] / (self.TP[-1] + self.FN[-1]) if (self.TP[-1] + self.FN[-1]) > 0 else 1)
            self.spe.append(self.TN[-1] / (self.TN[-1] + self.FP[-1]) if (self.TN[-1] + self.FP[-1]) > 0 else 0)
            self.comspe.append(self.FP[-1] / (self.TN[-1] + self.FP[-1]) if (self.TN[-1] + self.FP[-1]) > 0 else 0)
            self.pre.append(self.TP[-1] / (self.TP[-1] + self.FP[-1]) if (self.TP[-1] + self.FP[-1]) > 0 else 1)
            self.TPTN.append(self.TP[-1] + self.TN[-1])
            self.structsim.append(self.TPTN[-1] / self._npl)
            n = 2 * self.TP[-1] + self.FP[-1] + self.FN[-1]
            self.F1.append(2 * self.TP[-1] / n if n > 0 else 0)
            n = (self.TP[-1] + self.FP[-1]) * (self.TP[-1] + self.FN[-1]) * (self.TN[-1] + self.FP[-1]) * (self.TN[-1] + self.FN[-1])
            self.MCC.append((self.TP[-1] * self.TN[-1] - self.FP[-1] * self.FN[-1]) / np.sqrt(n) if n > 0 else 0)

    def signGraph_measures(self, Alist):
        """Calculate signed graph measures."""
        for i in range(Alist.shape[2]):
            T = Alist[:, :, i]
            STT = np.zeros_like(self._A)
            STT[T > self.tol] = 1
            STT[T < -self.tol] = -1
            self.TR.append(np.sum((self._STA == 1) & (STT == 1)) + np.sum((self._STA == -1) & (STT == -1)))
            self.TZ.append(np.sum((self._STA == 0) & (STT == 0)))
            self.FI.append(np.sum((self._STA == 1) & (STT == -1)) + np.sum((self._STA == -1) & (STT == 1)))
            self.FR.append(np.sum((self._STA == 0) & (np.abs(STT) > 0)))
            self.FZ.append(np.sum((np.abs(self._STA) > 0) & (STT == 0)))
            self.dirsen.append(self.TR[-1] / (self.TR[-1] + self.FI[-1] + self.FZ[-1]) if (self.TR[-1] + self.FI[-1] + self.FZ[-1]) > 0 else 0)
            self.dirspe.append(self.TZ[-1] / (self.TZ[-1] + self.FR[-1]) if (self.TZ[-1] + self.FR[-1]) > 0 else 0)
            self.dirprec.append(self.TR[-1] / (self.TR[-1] + self.FI[-1] + self.FR[-1]) if (self.TR[-1] + self.FI[-1] + self.FR[-1]) > 0 else 0)
            n = (self.TR[-1] + self.FR[-1]) * (self.TR[-1] + self.FI[-1] + self.FZ[-1]) * (self.TZ[-1] + self.FR[-1]) * (self.TZ[-1] + self.FI[-1] + self.FZ[-1])
            self.SMCC.append((self.TR[-1] * self.TZ[-1] - self.FR[-1] * (self.FI[-1] + self.FZ[-1])) / np.sqrt(n) if n > 0 else 0)

    def compare(self, Alist, selected=None):
        """Compare a 3D array of networks to the true network."""
        if self._A is None:
            raise ValueError("True network, A, needs to be set")
        Alist = np.asarray(Alist)
        if Alist.ndim != 3:
            Alist = Alist[:, :, np.newaxis]

        if self.issquare():
            self.system_measures(Alist)
            self.topology_measures(Alist)
            self.correlation_measures(Alist)
            self.graph_measures(Alist)
            self.signGraph_measures(Alist)
        else:
            # Assuming gsUtilities.rmdiag is not available; remove diagonal manually
            Alist_no_diag = Alist.copy()
            for i in range(Alist.shape[2]):
                np.fill_diagonal(Alist_no_diag[:, :, i], 0)
            self.topology_measures(Alist_no_diag)
            self.correlation_measures(Alist_no_diag)
            self.graph_measures(Alist_no_diag)
            self.signGraph_measures(Alist_no_diag)

        if selected is not None:
            allprops = self.show()
            results = {}
            for i in selected:
                results[allprops[i - 1]] = getattr(self, allprops[i - 1])  # 1-based to 0-based index
            return results
        return self

    def show(self, measure=None):
        """Get a list of measures or index of a specific measure."""
        allprops = [p for p in vars(self) if not p.startswith('_') and p not in ['tol', 'zetavec']]
        if measure is None:
            return allprops
        if isinstance(measure, str):
            return allprops.index(measure) + 1  # 1-based indexing for compatibility
        if isinstance(measure, (int, float)):
            return allprops[int(measure) - 1]  # 1-based to 0-based

    def max(self, measure='all'):
        """Find maximum values over measures."""
        props = self.show()
        if isinstance(measure, (int, float)):
            measure = props[int(measure) - 1]
        maximums = []
        maxind = []
        if measure == 'all':
            for prop in props:
                vals = np.array(getattr(self, prop))
                max_val, idx = np.max(vals), np.argmax(vals)
                maximums.append(max_val)
                maxind.append(idx + 1)  # 1-based index
            return np.array(maximums), np.array(maxind)
        else:
            ind = props.index(measure)
            vals = np.array(getattr(self, props[ind]))
            _, idx = np.max(vals), np.argmax(vals)
            maximums = [np.array(getattr(self, prop))[idx] for prop in props]
            return np.array(maximums), idx + 1

    def maxmax(self, *measures):
        """Return a new CompareModels object with max values."""
        T = CompareModels()
        props = self.show()
        if not measures:
            maxes, _ = self.max()
            for i, prop in enumerate(props):
                setattr(T, prop, [maxes[i]])
        else:
            for mess in measures:
                maxes, _ = self.max(mess)
                for i, prop in enumerate(props):
                    getattr(T, prop).append(maxes[i])
        return T

    def min(self, measure='all'):
        """Find minimum values over measures."""
        props = self.show()
        if isinstance(measure, (int, float)):
            measure = props[int(measure) - 1]
        minimums = []
        minind = []
        if measure == 'all':
            for prop in props:
                vals = np.array(getattr(self, prop))
                min_val, idx = np.min(vals), np.argmin(vals)
                minimums.append(min_val)
                minind.append(idx + 1)
            return np.array(minimums), np.array(minind)
        else:
            ind = props.index(measure)
            vals = np.array(getattr(self, props[ind]))
            _, idx = np.min(vals), np.argmin(vals)
            minimums = [np.array(getattr(self, prop))[idx] for prop in props]
            return np.array(minimums), idx + 1

    def minmin(self, *measures):
        """Return a new CompareModels object with min values."""
        T = CompareModels()
        props = self.show()
        if not measures:
            mins, _ = self.min()
            for i, prop in enumerate(props):
                setattr(T, prop, [mins[i]])
        else:
            for mess in measures:
                mins, _ = self.min(mess)
                for i, prop in enumerate(props):
                    getattr(T, prop).append(mins[i])
        return T

    def __add__(self, other):
        """Add measures from another CompareModels object."""
        props = self.show()
        pM = CompareModels(self._A)
        for prop in props:
            setattr(pM, prop, [x + y for x, y in zip(getattr(self, prop), getattr(other, prop))])
        return pM

    def vertcat(self, *others):
        """Vertically concatenate measures."""
        props = self.show()
        for N in others:
            for prop in props:
                getattr(self, prop).extend(getattr(N, prop))
        return self

    def horzcat(self, *others):
        """Horizontally concatenate measures."""
        return self.vertcat(*others)  # Same as vertcat for list-based storage

    def transpose(self):
        """Transpose all measures (no-op for lists)."""
        return self  # Lists don’t need transposing in this context

    def stack(self, *others):
        """Stack measures in 3rd dimension (simulated with lists)."""
        props = self.show()
        for N in others:
            for prop in props:
                getattr(self, prop).extend(getattr(N, prop))
        return self

    def mean(self):
        """Calculate mean of all measures."""
        props = self.show()
        mM = CompareModels(self._A)
        for prop in props:
            setattr(mM, prop, [np.mean(getattr(self, prop))])
        return mM

    def var(self):
        """Calculate variance of all measures."""
        props = self.show()
        vM = CompareModels(self._A)
        for prop in props:
            setattr(vM, prop, [np.var(getattr(self, prop))])
        return vM

    def getIndex(self, index):
        """Return object with measures at specified index."""
        T = CompareModels()
        props = self.show()
        index = np.array(index) - 1  # 1-based to 0-based
        for prop in props:
            vals = np.array(getattr(self, prop))
            setattr(T, prop, [vals[i] for i in index])
        return T

    def reshape(self, *shape):
        """Reshape measures (simulated for lists)."""
        T = CompareModels()
        props = self.show()
        for prop in props:
            vals = np.array(getattr(self, prop)).reshape(*shape)
            setattr(T, prop, vals.tolist())
        return T

    def issquare(self):
        """Check if A is square."""
        return self._A.shape[0] == self._A.shape[1] if self._A is not None else False

    def ROC(self):
        """Plot ROC curve and return AUROC."""
        auroc, TPR, FPR = self.AUROC()
        plt.figure()
        plt.plot(FPR, TPR)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve\nAUROC = {auroc:.3f}')
        plt.grid(True)
        plt.show()
        return auroc

    def AUROC(self):
        """Calculate area under ROC curve."""
        TPR = np.array(self.sen)
        FPR = np.array(self.comspe)
        ord = np.argsort(self.nlinks)
        TPR = TPR[ord]
        FPR = FPR[ord]
        auroc = np.trapz(TPR, FPR)
        return auroc, TPR, FPR

    def AUPR(self):
        """Calculate area under precision-recall curve."""
        PRE = np.array(self.pre)
        REC = np.array(self.sen)
        ord = np.argsort(self.nlinks)
        PRE = PRE[ord]
        REC = REC[ord]
        aupr = np.trapz(PRE, REC)
        return aupr, PRE, REC

    def PR(self):
        """Plot precision-recall curve and return AUPR."""
        aupr, PRE, REC = self.AUPR()
        plt.figure()
        plt.plot(REC, PRE)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR curve\nAUPR = {aupr:.3f}')
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.grid(True)
        plt.show()
        return aupr

    def save(self, savepath='dummy.tsv', fending=None):
        """Save performance measures to a file or return as table."""
        supported_json = {'.json', '.ubj'}
        supported_table = {'.txt', '.csv', '.dat', '.tsv', '.xls', '.xlsm', '.xlsx', '.xlsb'}

        if not fending:
            _, _, ext = os.path.splitext(savepath)
            fending = ext if ext else '.tsv'

        if not fending.startswith('.'):
            fending = '.' + fending

        p, f = os.path.split(savepath)
        if not p:
            p = './'
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Save path does not exist: {p}")
        savepath = os.path.join(p, f + (fending if not f.endswith(fending) else ''))

        if fending not in supported_json and fending not in supported_table:
            raise ValueError("File extension or format not supported")

        if fending in supported_table:
            measures = self.show()
            s = {m: getattr(self, m) for m in measures if getattr(self, m)}
            df = pd.DataFrame(s)
            if fending == '.tsv':
                df.to_csv(savepath, sep='\t', index=True)
            else:
                df.to_csv(savepath, index=True)  # Adjust for other formats as needed
            return df
        else:
            raise NotImplementedError("JSON/UBJ saving requires external libraries (e.g., json, ubjson)")
