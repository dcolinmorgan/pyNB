# pyGS — Python GeneSpider

Gene regulatory network inference and benchmarking, powered by [sparselink](sparselink/).

pyGS provides biology-specific workflows (data loading, TF filtering, gold-standard evaluation) on top of **sparselink**, a domain-agnostic sparse network inference library with 20+ methods behind a unified interface.

## Installation

```bash
# With uv (recommended)
uv pip install -e ".[dev]" -e "sparselink/[dev]"

# Or with pip
pip install -e ".[dev]"
pip install -e "sparselink/[dev]"

# Optional extras
uv pip install -e "sparselink/[causal]"   # PC, FCI (causal-learn)
uv pip install -e "sparselink/[deep]"     # DAG-GNN (torch)
```

Requires Python ≥ 3.11 (pyGS) or ≥ 3.10 (sparselink standalone).

## Quick Start

### Infer a network

```python
import numpy as np
from sparselink import get_method, list_methods

print(list_methods())
# ['lasso', 'elastic_net', 'lsco', 'clr', 'genie3', 'tigress',
#  'graphical_lasso', 'glasso_stars', 'pc', 'fci', 'notears', ...]

X = np.random.randn(100, 10)
result = get_method("lasso")(alpha=0.1).fit(X)
print(result.adjacency_matrix.shape)  # (10, 10)
```

### Benchmark with synthetic data

```python
from sparselink import get_method
from sparselink.bench import generate_network, generate_expression, evaluate

A_true = generate_network(n_genes=20, topology="scalefree")
X = generate_expression(A_true, n_samples=100, snr=10.0)

result = get_method("glasso")().fit(X)
metrics = evaluate(A_true, result.adjacency_matrix)
print(f"AUROC={metrics.auroc:.3f}  F1={metrics.f1:.3f}  MCC={metrics.mcc:.3f}")
```

### pygs CLI

```bash
pygs                                              # interactive mode
pygs status                                       # system info & available methods
pygs methods                                      # list all inference methods

# Inference
pygs infer data.csv -m lasso -o adj.npy           # infer a network

# Benchmarking
pygs bench --tier fast --timeout 60               # synthetic benchmark (sparselink)
pygs bench-gs --tier fast --sizes N50             # GeneSpider benchmark (real data)

# NestBoot FDR
pygs nestboot expr.csv -m lasso --fdr 0.05        # bootstrap FDR-controlled inference
pygs nestboot expr.h5ad -m elastic_net             # works with h5ad, csv, tsv, npy

# Evaluation & visualization
pygs evaluate pred.npy --gold gold.npy            # evaluate against gold standard
pygs plot adj.npy --genes genes.txt --tfs tfs.txt # plot GRN with TF highlighting

# Results
pygs dashboard -i results.json                    # HTML dashboard
pygs show results.json                            # render result table
```

### NestBoot FDR control

NestBoot performs bootstrap-based false discovery rate control for network inference.
It runs an inference method repeatedly on bootstrapped samples and compares against
shuffled-data baselines to identify statistically supported edges.

```python
from methods.nestboot import Nestboot
from config import AnalysisConfig

config = AnalysisConfig(total_runs=500, inner_group_size=10, fdr_threshold=0.05)
nb = Nestboot(config)
results = nb.run_nestboot(dataset, inference_method=my_method, nest_runs=50, boot_runs=10)
# results.xnet  — FDR-controlled adjacency matrix
# results.sxnet — signed network with edge direction
```

## Supported Methods

| Category | Methods |
|----------|---------|
| Regression | Lasso, Elastic Net, Ridge, LSCO, TIGRESS |
| Tree-based | GENIE3 |
| Information theory | CLR |
| Graphical models | Graphical Lasso, GLASSO+StARS, Neighborhood Selection |
| Correlation | Partial Correlation |
| Causal (time-series) | PCMCI, Granger, Transfer Entropy |
| Constraint-based | PC, FCI |
| Continuous optimization | NOTEARS, DAG-GNN |
| Bayesian | BDeu, BGe |

All methods implement `InferenceMethod.fit(X) -> InferenceResult`.

## Project Structure

```
pyGS/
├── src/                     # pyGS package (biology layer)
│   ├── methods/             # Method wrappers
│   ├── datastruct/          # Network, Dataset, Experiment
│   ├── analyze/             # CompareModels, Data loading
│   ├── bio/                 # Biology-specific workflows
│   └── bootstrap/           # NB-FDR analysis
├── sparselink/              # Standalone inference library
│   └── src/sparselink/
│       ├── methods/         # 20+ registered inference methods
│       ├── bench/           # Benchmarking, TUI, dashboard, NestBoot
│       └── accel.py         # MLX acceleration (Apple Silicon)
├── tests/                   # pyGS tests
├── docs/site/               # MkDocs documentation source
└── pyproject.toml
```

## Development

```bash
git clone https://github.com/dcolinmorgan/pyGS.git
cd pyGS
uv pip install -e ".[dev]" -e "sparselink/[dev]"

# Lint & format
ruff check src/ sparselink/
ruff format src/ sparselink/

# Type check
mypy

# Test
pytest --cov
```

## Documentation

```bash
uv pip install -e ".[docs]"
mkdocs serve
```

See [docs/README.md](docs/README.md) for extended usage examples and TUI walkthrough.

## License

MIT
