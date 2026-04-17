# Quickstart

## Installation

```bash
pip install pyGS
```

For optional dependencies:

```bash
pip install "pyGS[causal]"   # PC, FCI (causal-learn)
pip install "pyGS[deep]"     # DAG-GNN (torch)
pip install "pyGS[docs]"     # documentation build tools
```

sparselink installs automatically as a dependency. To use it standalone:

```bash
pip install sparselink
```

## Basic Usage

### Infer a network with sparselink

```python
import numpy as np
from sparselink import get_method, list_methods

# See all available methods
print(list_methods())
# ['lasso', 'partial_correlation', 'lsco', 'clr', 'elastic_net', 'ridge',
#  'pcmci', 'granger', 'transfer_entropy', 'graphical_lasso', 'glasso_stars',
#  'neighborhood_selection', 'genie3', 'tigress', 'pc', 'fci', 'notears',
#  'dag_gnn', 'bdeu', 'bge']

# Generate sample data (100 samples, 10 features)
X = np.random.randn(100, 10)

# Run inference
Method = get_method("lasso")
result = Method(alpha=0.1).fit(X)

# Access results
print(result.adjacency_matrix.shape)  # (10, 10)
print(result.edge_list[:5])           # [(src, tgt, weight), ...]
```

### Run a benchmark

```python
from sparselink.bench import generate_network, generate_expression, evaluate

# Create synthetic ground truth
A_true = generate_network(n_genes=20, topology="scalefree")
X = generate_expression(A_true, n_samples=100, snr=2.0)

# Infer and evaluate
Method = get_method("glasso_stars")
result = Method().fit(X)
metrics = evaluate(result.adjacency_matrix, A_true)
print(metrics)  # {'auroc': 0.85, 'aupr': 0.72, ...}
```

### NestBoot FDR control

```python
from sparselink.bench import NestBoot

nestboot = NestBoot(method_name="lasso", n_bootstraps=50, fdr=0.05)
final_adj = nestboot.run(X)
# Returns thresholded adjacency with FDR control
```

### pyGS biology workflow

```python
from bio.preprocessing import load_expression_matrix, filter_tf_targets
from bio.evaluation import compare_to_gold_standard

# Load and filter
expr = load_expression_matrix("expression.csv")
tf_expr, target_expr = filter_tf_targets(expr, tf_list="tfs.txt")

# Infer network (uses sparselink under the hood)
from methods import run
results = run(tf_expr.values, method="lasso", thresholds=[0.01, 0.05, 0.1])

# Compare to gold standard
metrics = compare_to_gold_standard(results[:, :, 0], "gold_standard.csv")
```

## CLI

The `pygs` command provides a unified CLI for pyGS workflows. Running `pygs` with no arguments launches interactive mode.

```bash
pygs status                                       # system info
pygs methods                                      # list inference methods
pygs infer data.csv -m lasso                      # infer a GRN
pygs bench --tier fast                            # synthetic benchmark
pygs bench-gs --tier fast --sizes N50             # GeneSpider benchmark
pygs nestboot data.csv -m lasso                   # NestBoot FDR analysis
pygs evaluate pred.npy --gold gold.npy            # evaluate against gold standard
pygs plot pred.npy --genes genes.txt              # plot a GRN
pygs dashboard -i results.json                    # HTML dashboard
pygs show results.json                            # render result table
```

sparselink also has its own TUI:

```bash
sparselink-tui
```
