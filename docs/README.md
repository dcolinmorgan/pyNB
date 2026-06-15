# pyGS — Python GeneSpider

Gene regulatory network inference and benchmarking, powered by [sparselink](https://pypi.org/project/sparselink/).

## Installation

```bash
uv pip install -e ".[dev]"

# Optional extras
uv pip install -e ".[netzoo]"     # PANDA (netZooPy)
uv pip install -e ".[scenic]"    # SCENIC+ (pycisTopic)
uv pip install -e ".[causal]"    # PC, FCI (causal-learn)
uv pip install -e ".[deep]"      # DAG-GNN (torch)
```

sparselink (21 inference methods) is installed automatically from [PyPI](https://pypi.org/project/sparselink/).

## Quick Start

### Interactive TUI

```bash
pygs-tui                          # interactive mode
pygs-tui infer data.h5ad -m genie3 --directed   # infer + direction
pygs-tui bench-gs                 # GeneSpider benchmark
pygs-tui status                   # show methods & deps
```

### Python API

```python
from sparselink import get_method, list_methods
import numpy as np

X = np.random.randn(100, 50)
result = get_method("ensemble")().fit(X)  # Borda-count consensus
print(result.adjacency_matrix.shape)      # (50, 50)
```

### Benchmark with GeneSpider data

```bash
pygs-tui bench-gs
# Select: lasso, scenicplus, 4   → runs both + NestBoot comparison
```

## Supported Methods

### sparselink (domain-agnostic, from PyPI)

| Category | Methods |
|----------|---------|
| Regression | Lasso, LassoCV, Elastic Net, Ridge, LSCO, TIGRESS |
| Tree-based | GENIE3 |
| Information theory | CLR, PIDC |
| Graphical models | Graphical Lasso, GLASSO+StARS, Neighborhood Selection |
| Correlation | Partial Correlation |
| Causal (time-series) | PCMCI, Granger, Transfer Entropy |
| Constraint-based | PC, FCI |
| Continuous optimization | NOTEARS, DAG-GNN |
| Bayesian | BDeu, BGe |
| Ensemble | Borda-count consensus |

### pyGS (biology-specific)

| Category | Methods |
|----------|---------|
| Integrative GRN | PANDA (netZooPy), SCENIC+ |
| Perturbation-response | SINCERITIES, DSPIN |
| Time-series GRN | GRISLI, LEAP |
| Prior-guided | Prior-LASSO |

pyGS methods are auto-discovered — just drop a `.py` in `src/methods/`.

## Features

- **NestBoot FDR** — bootstrap-based false discovery rate control
- **Edge direction** — infer causality from perturbation asymmetry
- **h5ad/CSV/TSV/NPY input** — load any expression format
- **Interactive dashboard** — click-to-drill HTML visualization
- **MLX acceleration** — Apple Silicon GPU for matrix ops
- **Auto-discovery** — add methods without editing config files

## Project Structure

```
pyGS/
├── src/
│   ├── methods/          # Bio-specific method wrappers + auto-discovery registry
│   ├── bio/              # direction inference, data I/O, preprocessing
│   ├── bench/            # CLI, GeneSpider benchmark, TUI, NestBoot integration
│   ├── datastruct/       # Network, Dataset classes
│   ├── analyze/          # CompareModels, Data loading
│   └── bootstrap/        # NB-FDR core
├── tests/                # Verification scripts
├── docs/                 # Documentation
└── pyproject.toml
```

## Development

```bash
git clone https://github.com/dcolinmorgan/pyGS.git
cd pyGS
uv pip install -e ".[dev]"

ruff check src/
mypy
pytest --cov
```

## License

MIT
