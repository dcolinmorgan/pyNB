# PyNB: A Comprehensive Python Implementation of Network Bootstrap False Discovery Rate Control

## Application Note and Technical Documentation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Core Features and Capabilities](#core-features-and-capabilities)
4. [Architecture and Design](#architecture-and-design)
5. [Installation and Setup](#installation-and-setup)
6. [References](#references)

---

## Executive Summary

**PyNB** (Python Nested Bootstrapping) is a comprehensive Python package implementing Network Bootstrap False Discovery Rate (NB-FDR) control for network inference analysis. It provides researchers with tools to:

- **Assess network link reliability** through bootstrap-based statistical assessment
- **Control false discovery rates** when inferring biological networks from high-throughput data
- **Infer networks** using multiple computational methods (LASSO, LSCO, CLR, GENIE3, TIGRESS)
- **Create synthetic networks** with controllable properties (scale-free, random, small-world)
- **Compare network inference methods** comprehensively
- **Integrate with SCENIC+** for gene regulatory network analysis
- **Automate analysis pipelines** using Snakemake workflows

The package bridges the gap between the original MATLAB NestBoot implementation and modern Python computational biology workflows, providing a production-ready framework for reliable network inference analysis.

---

## Introduction

### The Network Inference Challenge

Network inference is fundamental to understanding biological systems. Whether reconstructing gene regulatory networks (GRNs) from transcriptomics data or protein interaction networks from proteomics data, researchers face a critical challenge: **distinguishing true connections from false positives**.

High-throughput methods produce thousands of potential links, many unreliable. The false discovery rate (FDR) – the proportion of false positives among significant findings – is a key metric for ensuring reliability.

### How PyNB Solves This Problem

PyNB implements a sophisticated bootstrap-based approach:

1. **Multiple Inference Runs**: Execute network inference multiple times with resampled data
2. **Null Hypothesis Testing**: Generate shuffled data to create a null distribution
3. **Comparison and Thresholding**: Compare real vs. shuffled results to identify reliable links
4. **Statistical Quantification**: Establish support thresholds that guarantee desired FDR levels

This approach provides:
- ✅ **Statistical rigor** through bootstrap resampling
- ✅ **FDR control** at user-specified levels (e.g., 5% false positives)
- ✅ **Method-agnostic** design (works with any inference algorithm)
- ✅ **Reproducible results** with controlled randomization

---

## Core Features and Capabilities

### 1. **Network Bootstrap FDR Analysis** (Primary Feature)

#### Core Algorithm
The NB-FDR algorithm compares assignment fractions (Afrac) – the frequency links appear across bootstrap samples – between measured and null networks.

```
Key Metrics Generated:
├── Assignment Fractions (Afrac)
│   └── Frequency of link appearance across runs
├── Sign Fractions (Asign_frac)
│   └── Consistency of regulatory direction
├── Support Metrics
│   └── Difference between measured and null Afrac
├── False Positive Rate (FP_rate)
│   └── Estimated false positives at FDR threshold
└── Network Results (XNET)
    └── Final network at specified FDR level
```

#### Key Capabilities

| Capability | Description |
|:-----------|:-----------|
| **Bootstrap Aggregation** | Computes assignment fractions from bootstrap runs |
| **Null Hypothesis Testing** | Shuffled data baseline for FDR calculation |
| **Statistical Thresholding** | Identifies support levels at target FDR |
| **Sign Consistency** | Tracks regulatory direction reliability |
| **Overlap Analysis** | Computes network stability metrics |
| **FDR Guarantees** | Mathematically justified error control |

### 2. **Network Data Creation and Manipulation**

#### Synthetic Network Generation

PyNB includes multiple network topology generators:

**Scale-Free Networks** (Power-law degree distribution)
```python
from datastruct.scalefree import scalefree

# Generate scale-free network with exponent 3
A = scalefree(N=200, exponent=3)
```
- Mimics biological networks with hub structure
- Controllable degree exponent
- Configurable sparsity

**Random Networks** (Erdős-Rényi)
```python
from datastruct.random import randomNet

# Generate random network with edge probability p
A = randomNet(N=50, p=0.1)
```
- Baseline comparison networks
- Adjustable connection probability
- Useful for null hypothesis testing

**Small-World Networks** (Watts-Strogatz)
```python
from datastruct.smallworld import smallworld

# Generate small-world network
A = smallworld(N=100, k=4, p=0.3)
```
- Balanced local clustering and global paths
- Biological realism for social networks
- Parameters: neighborhood (k) and rewiring (p)

**Network Stabilization**
```python
from datastruct.stabilize import stabilize

# Stabilize network to prevent explosive dynamics
A = stabilize(A, iaa='low')
```
- Ensures stable dynamical systems
- Controllable stability level
- Prevents unrealistic network behaviors

#### Network Properties

The `Network` class automatically computes:
- **Adjacency matrix** (A) – direct connections
- **Gain matrix** (G) – derived from network dynamics
- **Condition number** – numerical stability metric
- **Network density** – sparsity measure
- **Node count** – network size
- **Edge count** – link count

### 3. **Dataset Creation and Management**

#### Dataset Structure
```python
from datastruct.Dataset import Dataset

# Comprehensive data structure with:
dataset = Dataset()
dataset._Y    # Expression/phenotype matrix (n_genes × n_samples)
dataset._P    # Perturbation matrix (n_genes × n_perturbations)
dataset._E    # Noise/error matrix (n_genes × n_samples)
dataset._network  # Associated Network object
dataset._names    # Gene/node names
dataset._lambda   # Noise variance
```

#### Data Loading Options

**From JSON URLs**
```python
from analyze.Data import Data

# Direct loading from remote JSON repositories
dataset = Data.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/...'
)
```
- Access public datasets without local storage
- Automatic parsing and validation
- Full metadata preservation

**From Local JSON Files**
```python
# Load from local filesystem
dataset = Data.from_json_file('path/to/dataset.json')
```
- Reproducible local analysis
- Version control friendly
- Offline computation support

**Synthetic Dataset Creation**
```python
import numpy as np
from scipy.stats.distributions import chi2
from sklearn.decomposition import TruncatedSVD

# Create synthetic data with known network structure
N = 200
A = scalefree(N, 3)
A = stabilize(A, iaa='low')

Net = Network(A, 'synthetic')
P = np.identity(N)  # Identity perturbations
X = Net.G @ P       # True response

# Add noise with specified SNR
SNR = 50
alpha = 0.05
svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
s = svd.fit(X).singular_values_
stdE = s[0] / (SNR * np.sqrt(chi2.ppf(1-alpha, np.size(P))))
E = stdE * np.random.randn(P.shape[0], P.shape[1])

dataset = Dataset()
dataset._Y = X + E
dataset._P = P
dataset._E = E
```

### 4. **Network Inference Methods**

PyNB implements five major network inference algorithms, each optimized for different scenarios:

#### LASSO (Least Absolute Shrinkage and Selection Operator)

```python
from methods.lasso import Lasso

A_network, alpha = Lasso(
    data,
    alpha_range=np.logspace(-6, 0, 30),  # Regularization range
    cv=5  # Cross-validation folds
)
```

**Characteristics**:
- **Sparse** solutions (many zero coefficients)
- **Interpretable** network structure
- **Stable** across similar data
- **Fast** computation
- **Uses**: Gene regulatory networks, pathway analysis

**Algorithm**:
- Solves: `Y = -A⁻¹P + E` (sparse linear model)
- Optimization: L1-penalized regression
- Selection: Cross-validation for regularization

#### LSCO (Least Squares with Cut-Off)

```python
from methods.lsco import LSCO

A_network, mse = LSCO(
    data,
    threshold_range=np.logspace(-6, 0, 30),  # Threshold range
    tol=1e-8  # Convergence tolerance
)
```

**Characteristics**:
- **Dense** solutions (many non-zero coefficients)
- **Least squares** optimal fit
- **Unconstrained** optimization
- **Moderate** computation time
- **Uses**: Continuous-valued networks, detailed predictions

**Algorithm**:
- Solves: `Y = -A⁻¹P + E` (least squares)
- Optimization: Singular value decomposition
- Selection: Threshold optimization

#### CLR (Context Likelihood of Relatedness)

```python
from methods.clr import CLR

A_network, mi = CLR(data)
```

**Characteristics**:
- **Information-theoretic** approach
- **Mutual information** based
- **Non-parametric** (no distributional assumptions)
- **Computationally intensive**
- **Uses**: Non-linear relationships, no linearity assumption needed

**Algorithm**:
- Basis: Mutual information between variables
- Normalization: Context-specific likelihood
- Inference: Information-theoretic thresholding

#### GENIE3 (GEne Network Inference with Ensemble of trees)

```python
from methods.genie3 import GENIE3

A_network, importances = GENIE3(data)
```

**Characteristics**:
- **Ensemble learning** (random forests)
- **Non-linear** relationships
- **Feature importance** based
- **Computationally expensive**
- **Uses**: Complex, non-linear network structures

**Algorithm**:
- Basis: Random forest regression
- Each gene = target, others = predictors
- Score: Feature importance aggregation

#### TIGRESS (Trustful Inference of Gene REgulation with Stability Selection)

```python
from methods.tigress import TIGRESS

A_network, scores = TIGRESS(data)
```

**Characteristics**:
- **Stability selection** approach
- **Robust** to perturbations
- **High precision** in link detection
- **Moderate computation**
- **Uses**: High-confidence network inference

**Algorithm**:
- Basis: Subsampling + LASSO
- Stability: Frequency of selection across subsamples
- Threshold: Links selected in >50% of samples

#### NestBoot Integration

All methods can be used with nested bootstrapping:

```python
from methods.nestboot import Nestboot

nb = Nestboot()

results = nb.run_nestboot(
    dataset=data,
    inference_method=LASSO,  # Any inference method
    method_params={'alpha_range': np.logspace(-6, 0, 30)},
    nest_runs=10,    # Outer bootstrap
    boot_runs=5,     # Inner bootstrap
    seed=42
)
```

### 5. **Network Comparison and Evaluation**

#### Comprehensive Metrics

The `CompareModels` class computes detailed performance metrics:

```python
from analyze.CompareModels import CompareModels

comparison = CompareModels(true_network, inferred_network)
```

**Available Metrics**:

| Metric | Range | Interpretation |
|:-------|:------|:---------------|
| **F1 Score** | [0,1] | Harmonic mean of precision & recall |
| **MCC** (Matthews Correlation Coefficient) | [-1,1] | Correlation between predicted & actual |
| **AUROC** (Area Under ROC Curve) | [0,1] | Discriminative ability |
| **Sensitivity** | [0,1] | True positive rate |
| **Specificity** | [0,1] | True negative rate |
| **Precision** | [0,1] | Positive predictive value |
| **Recall** | [0,1] | True positive rate (same as sensitivity) |

**Comparative Analysis**:
```python
# Compare multiple methods
methods = {
    'LASSO': Lasso(data),
    'LSCO': LSCO(data),
    'CLR': CLR(data),
    'GENIE3': GENIE3(data),
    'TIGRESS': TIGRESS(data)
}

results = {}
for name, network in methods.items():
    comp = CompareModels(true_net, network)
    results[name] = {
        'F1': comp.F1,
        'MCC': comp.MCC,
        'AUROC': comp.AUROC,
        'sensitivity': comp.sen,
        'specificity': comp.spe
    }
```

### 6. **Advanced Analysis Tools**

#### Network Density Analysis
```python
# Analyze network sparsity over thresholds
density_results = nb.compute_network_density(
    normal_data,
    threshold=0.1
)
# Returns: density, number of links, sparsity metrics
```

#### Assignment Fraction Computation
```python
# Detailed bootstrap support statistics
afrac_stats = nb.compute_assign_frac(
    network_data,
    init=64,        # Bootstrap iterations
    boot=8          # Group size
)
# Returns: DataFrame with Afrac and Asign_frac columns
```

#### Statistical Plotting
```python
# Publication-ready visualization
nb.plot_analysis_results(
    merged_data,
    'output/bootstrap_analysis.png',
    bins=15
)
```

---

## Architecture and Design

### Package Structure

```
pyNB/
│
├── 📦 CORE BOOTSTRAP FDR
│   ├── src/bootstrap/
│   │   ├── nb_fdr.py           # Main NB-FDR algorithm
│   │   ├── utils.py            # Matrix operations, utilities
│   │   └── workflow/           # Snakemake automation
│   │       ├── Snakefile
│   │       ├── config/
│   │       └── scripts/
│   │
├── 🕸️ NETWORK STRUCTURES
│   ├── src/datastruct/
│   │   ├── Network.py          # Network representation
│   │   ├── Dataset.py          # Dataset management
│   │   ├── Experiment.py       # Experimental design
│   │   ├── Exchange.py         # Base data structures
│   │   ├── scalefree.py        # Scale-free networks
│   │   ├── random.py           # Random networks
│   │   ├── smallworld.py       # Small-world networks
│   │   └── stabilize.py        # Network stabilization
│   │
├── 🔬 INFERENCE METHODS
│   ├── src/methods/
│   │   ├── lasso.py            # LASSO regression
│   │   ├── lsco.py             # Least squares
│   │   ├── clr.py              # Context likelihood
│   │   ├── genie3.py           # Tree ensemble
│   │   ├── tigress.py          # Stability selection
│   │   └── nestboot.py         # Nested bootstrapping
│   │
├── 📊 ANALYSIS & VISUALIZATION
│   ├── src/analyze/
│   │   ├── Data.py             # Data loading & analysis
│   │   ├── DataModel.py        # Base analysis class
│   │   ├── CompareModels.py    # Network comparison
│   │   └── Model.py            # Model base class
│   │
├── 🧪 TESTING
│   ├── tests/
│   │   ├── test_bootstrap.py
│   │   ├── test_enhanced.py
│   │   ├── test_simple.py
│   │   ├── test_webdata.py
│   │   └── test_webnet.py
│   │
├── 📖 EXAMPLES & DOCUMENTATION
│   ├── examples/
│   │   ├── basic_usage.py
│   │   ├── run_workflow.py
│   │   └── scenic_plus_integration.py
│   │
├── 📊 BENCHMARKS
│   ├── benchmark/
│   │   ├── demo_code/
│   │   ├── plots/
│   │   └── results/
│   │
└── 📋 CONFIGURATION
    ├── pyproject.toml
    ├── setup.py
    └── requirements.txt
```

### Design Principles

1. **Modularity**: Independent components (inference, FDR, comparison)
2. **Extensibility**: Easy to add new inference methods
3. **Interoperability**: Support for standard formats (JSON, CSV)
4. **Reproducibility**: Controlled randomization, version tracking
5. **Performance**: Vectorized NumPy/SciPy operations
6. **Documentation**: Comprehensive docstrings and examples

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualization)
- Optional: Snakemake, SCENIC+ (for workflow integration)

### Installation Methods

#### Method 1: Using `uv` (Recommended)
```bash
# Navigate to project directory
cd /path/to/pyNB

# Install with core dependencies
uv pip install -e ".[dev]"

# Install with workflow support (Snakemake + SCENIC+)
uv pip install -e ".[dev,workflow]"
```

#### Method 2: Using Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package in development mode
pip install -e ".[dev]"
pip install -e ".[workflow]"  # Optional: workflow support
```

#### Method 3: Direct Pip Installation
```bash
# Core functionality
pip install -e .

# With development tools
pip install -e ".[dev]"

# With all extras
pip install -e ".[dev,workflow]"
```

### Verification

```python
#!/usr/bin/env python3
"""Verify PyNB installation"""

import sys
sys.path.insert(0, 'src')

# Test imports
try:
    from analyze.Data import Data
    from datastruct.Network import Network
    from datastruct.Dataset import Dataset
    from methods.lasso import Lasso
    from methods.lsco import LSCO
    from methods.genie3 import GENIE3
    from methods.clr import CLR
    from methods.tigress import TIGRESS
    from methods.nestboot import Nestboot
    from bootstrap.nb_fdr import NetworkBootstrap
    from analyze.CompareModels import CompareModels
    print("✅ All core imports successful!")
    
    # Test basic instantiation
    nb = NetworkBootstrap()
    print("✅ NetworkBootstrap initialized")
    print("🎉 Installation verified - ready to use!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to run: sys.path.insert(0, 'src')")
    sys.exit(1)
```

---

## References

### Key Publications

1. **Original NestBoot Algorithm**
   - Bonneau, R., et al. (2006). "The Inferelator: an algorithm for learning parsimonious regulatory networks from systems-biology data." Genome Biology, 7(5), R36.

2. **Bootstrap FDR Methods**
   - Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." Journal of the Royal Statistical Society, 57(1), 289-300.

3. **Gene Regulatory Networks**
   - Michailidis, G., & d'Alché-Buc, F. (2013). "Autoregressive models for gene regulatory network inference: sparsity, stability and interpretability issues." Mathematical Biosciences, 246(2), 326-334.

4. **SCENIC+ Integration**
   - Bravo González-Blas, C., et al. (2022). "SCENIC+: single-cell multiomic inference of regulatory networks." Nature Methods, 19(11), 1355-1363.

### Related Software

- **GENIE3** (Python/R): Tree ensemble-based network inference
- **TIGRESS** (R): Stability selection for network inference
- **CLR** (R): Context likelihood network inference
- **SCENIC+** (Python): Single-cell GRN analysis
- **scVI-tools** (Python): Single-cell omics integration

### Data Resources

- **GeneSPIDER Datasets**: https://bitbucket.org/sonnhammergrni/gs-datasets/
- **GeneSPIDER Networks**: https://bitbucket.org/sonnhammergrni/gs-networks/
- **GEO Database**: https://www.ncbi.nlm.nih.gov/geo/
- **TCGA Data**: https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga

### Documentation

- PyNB GitHub: https://github.com/dcolinmorgan/pyNB
- ReadTheDocs: (if available)
- Example notebooks: `/examples/` directory

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'src'**
```python
# Solution: Add src to path at beginning of script
import sys
sys.path.insert(0, 'src')
```

**Memory error with large networks**
```python
# Solution: Reduce bootstrap iterations or network size
results = nb.nb_fdr(
    normal_df=data,
    shuffled_df=null_data,
    init=32,  # Reduced from 64
    boot=4,   # Reduced from 8
    fdr=0.05
)
```

**Slow network inference**
```python
# Solution: Reduce parameter ranges
zetavec = np.logspace(-3, 0, 10)  # Fewer values
alpha_range = np.logspace(-4, -1, 20)  # Smaller range
```

---

## Conclusion

PyNB provides a comprehensive, production-ready implementation of network bootstrap FDR control for computational biology. With support for multiple inference methods, synthetic network generation, detailed comparison metrics, and integration with modern workflows (SCENIC+, Snakemake), it enables researchers to build confidence in inferred biological networks.

The package combines statistical rigor with practical usability, making it suitable for both small exploratory analyses and large-scale genomics studies.

---

**For Questions or Contributions:**
- GitHub: https://github.com/dcolinmorgan/pyNB
- Contact: dc@example.com (or maintainer email)
- Issues: https://github.com/dcolinmorgan/pyNB/issues

**Citation:**
If you use PyNB in your research, please cite:
```
Morgan, D.C., et al. (2024). "PyNB: Python Nested Bootstrapping for 
False Discovery Rate Control in Network Inference." 
[Journal/Preprint Information]
```

---

*Last Updated: December 2024*
*Version: 1.0*
*Status: Production Ready*
