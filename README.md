# Network Bootstrap FDR & mini GeneSPIDER

A comprehensive Python implementation of NB-FDR (Network Bootstrap False Discovery Rate) analysis for gene regulatory network inference and evaluation. This package implements an algorithm to estimate bootstrap support for network links by comparing measured networks against a shuffled (null) dataset.

## 🚀 Supported Methods

This package includes implementations of the following network inference methods:

1. **LASSO** (Least Absolute Shrinkage and Selection Operator)
2. **LSCO** (Least Squares with Cut-Off)
3. **CLR** (Context Likelihood of Relatedness)
4. **GENIE3** (GEne Network Inference with Ensemble of trees)
5. **TIGRESS** (Trustful Inference of Gene REgulation with Stability Selection)

## 📦 Installation

### Quick Start with uv (Recommended)
```bash
# Navigate to project directory
cd /path/to/pyNB

# For development with all features
uv pip install -e ".[dev,workflow]"
```

### Alternative Installation
```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate  
pip install -e ".[dev]"            # Core functionality + testing
pip install -e ".[workflow]"       # + Snakemake & SCENIC+ integration
```

## ⚡ Quick Start: Create a Dataset/Network

```python
import sys, numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.stats.distributions import chi2
sys.path.insert(0, 'src')

from analyze.Data import Data
from datastruct.Network import Network
from datastruct.scalefree import scalefree
from datastruct.random import randomNet
from datastruct.stabilize import stabilize
import analyze
from datastruct.Dataset import Dataset
N=200
A=scalefree(N,3)
A = stabilize(A, iaa='low')

Net = Network(A, 'myNetwork')

P=np.identity(N)

X = Net.G@P


SNR = 50
alpha=0.05
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
s = svd.fit(X).singular_values_
stdE = s[0]/(SNR*np.sqrt(chi2.ppf(1-alpha,np.size(P))))
E = stdE*np.random.randn(P.shape[0],P.shape[1])

F = np.zeros_like(P)

D= Dataset
D.network = Net.network
D.E = E
D.F = F 
D.Y = X+E
D.P = P
D.lambda_ = [stdE**2,0]
D.cvY = D.lambda_[0]*np.identity(N)
D.cvP = np.zeros(N)
D.sdY = stdE*np.ones(D.P.shape)
D.sdP = np.zeros(D.P.shape)

Data = Dataset(D, Net)
from methods.lsco import LSCO
from analyze.CompareModels import CompareModels

zeta = np.logspace(-6,0,30)
infMethod = 'LSCO'
[Aest0, z0] = LSCO(Data,zeta)
M = CompareModels(Net, Aest0)

M.AUROC
max(M.F1)

```

## ⚡ Quick Start: Basic Inference

Here is a simple example of how to load a dataset and run inference using various supported methods.

```python
import sys
import numpy as np
sys.path.insert(0, 'src')

from analyze.Data import Data
from datastruct.Network import Network
from analyze.CompareModels import CompareModels
from methods.lasso import Lasso
from methods.lsco import LSCO
from methods.genie3 import GENIE3
from methods.clr import CLR

dataset = Data.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR100000-IDY252384.json'
)
true_net = Network.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json'
)

# 2. Run Inference Methods

zetavec = np.logspace(-6, 0, 30)
lasso_net, _ = Lasso(dataset, alpha_range=zetavec)
lsco_net,_ = LSCO(dataset, threshold_range=zetavec)
genie3_net,_ = GENIE3(dataset)
clr_net,_ = CLR(dataset)

M0 = CompareModels(true_net,lasso_net)
print(M0.F1)
M1 = CompareModels(true_net,lsco_net)
print(M1.AUROC)
M1 = CompareModels(true_net,genie3_net)
print(M1.MCC)
M2 = CompareModels(true_net,clr_net)
print(M2.AUROC)

```

## 📊 Benchmark Results Visualization

After running the benchmark, you can generate comprehensive performance comparison plots using the included visualization notebooks.

### Sample Results

Here are example performance comparison plots for the 5 supported methods (N50 networks):

![MATLAB Comparison Violin Plot](benchmark/plots/language_comparison_all.png)


Each plot shows F1 Score, MCC, and AUROC comparisons across different SNR levels, providing a comprehensive view of method performance and stability.


### BENCHMARK CODE
The benchmark and plotting scripts are located in the [benchmark](benchmark/demo_code/n50_benchmark.ipynb). You can customize and extend these scripts to suit your analysis needs.

