# pyGS — Python GeneSpider

Gene regulatory network inference and benchmarking, powered by [sparselink](sparselink/).

## Overview

pyGS provides biology-specific workflows (data loading, TF filtering, gold-standard evaluation) on top of **sparselink**, a domain-agnostic sparse network inference library with 20 methods behind a unified interface.

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

## Quick Start

### Interactive TUI (recommended)

```bash
sparselink-tui                    # if installed
uv run -m sparselink.bench.tui    # without installing
```

Launches an interactive terminal UI where you can configure and run benchmarks, generate dashboards, and inspect results — all from a menu:

```
 ▄▄▄▄  ▄▄▄▄   ▄▄▄  ▄▄▄▄  ▄▄▄▄  ▄▄▄▄  █     ▀ ▄▄▄  █  ▄
 █▀▀▀  █▀ ▀█  █▀ █  █▀ ▀▄ █▀▀▀  █▀▀▀  █     █ █  █  █▄▀
 ▀▀▀█  █▀▀▀   █▀▀█  █▀▀▄  ▀▀▀█  █▀▀   █     █ █  █  █ ▀▄
 ▄▄▄█▀ █      █  █  █  █  ▄▄▄█▀ █▄▄▄  █▄▄▄▀ █ █  █  █  █

  1  Show system status & available methods
  2  Run synthetic benchmark
  3  Run GeneSpider benchmark
  4  Generate interactive HTML dashboard
  5  Render a previous result JSON

sparselink ❯
```

Option **2** walks you through configuring every dimension:
- **A)** Method tier (fast / medium / slow / very_slow)
- **B)** Network size (20 / 50 / 100 genes)
- **C)** Sparsity levels (0.2, 0.4, 0.6)
- **D)** SNR levels (0.1, 1.0, 10.0)
- **E)** Number of replicates

### TUI CLI commands

```bash
sparselink-tui status                             # check methods, MLX, deps
sparselink-tui bench --tier fast --timeout 60     # synthetic benchmark
sparselink-tui bench-gs --sizes N50               # GeneSpider benchmark
sparselink-tui show benchmark_results.json        # render results table
sparselink-tui dashboard -i results.json          # generate + open HTML dashboard
```

### Infer a network with sparselink

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

The `pygs` command provides a unified CLI for pyGS-specific workflows (GeneSpider benchmarks, biology-layer status). Running `pygs` with no arguments launches interactive mode:

```bash
pygs status                                       # system info
pygs methods                                      # list inference methods
pygs infer data.csv -m lasso                      # infer a GRN
pygs bench --tier fast                            # synthetic benchmark
pygs bench-gs --tier fast --sizes N50             # GeneSpider benchmark
pygs bench-gs --tier fast,nestboot --sizes N50    # GeneSpider direct vs NestBoot comparison
pygs nestboot data.csv -m lasso                   # NestBoot FDR analysis
pygs evaluate pred.npy --gold gold.npy            # evaluate against gold standard
pygs plot pred.npy --genes genes.txt              # plot a GRN
pygs dashboard -i results.json                    # HTML dashboard
pygs show results.json                            # render result table
pygs                                              # interactive mode
```

### Benchmark against real GeneSpider data

```bash
pygs bench-gs --tier fast --sizes N50 --timeout 120

# Enable NestBoot comparison via --tier (alternative to --nestboot flag)
pygs bench-gs --tier fast,nestboot --sizes N50 --timeout 120
```

Downloads datasets from the [Sonnhammer GRNi repos](https://bitbucket.org/sonnhammergrni/) and evaluates all methods with alpha sweep and perturbation matrix support.

You can enable NestBoot wrapping in two ways: pass `--nestboot` as a flag, or include `nestboot` in the `--tier` list. When `nestboot` is the only tier, methods default to the `fast` tier.

When NestBoot is enabled, each method is run twice — once direct and once with NestBoot wrapping — and results are rendered in a side-by-side comparison table.

### NestBoot FDR via CLI

Run bootstrap-based FDR analysis directly from the command line:

```bash
pygs nestboot expression.csv -m lasso                        # defaults: 10×10, FDR 0.05
pygs nestboot expression.csv -m elastic_net --nest 20 --boot 20 --fdr 0.01
pygs nestboot expression.h5ad -m glasso -o my_network.npy
pygs nestboot expression.csv -m genie3                       # post-hoc thresholding
```

NestBoot adapts its sparsity-sweep strategy based on the chosen method:

- Methods with a native regularization parameter (lasso, elastic_net, ridge, lsco, glasso, neighborhood_selection) are swept across a range of alpha values to produce the variation NestBoot needs.
- All other methods (e.g. genie3, clr, pc, notears) are fit once, then a post-hoc threshold sweep over the continuous adjacency scores generates the required sparsity variation.

Both strategies produce a 3D stack of adjacency matrices that NestBoot aggregates for FDR control. The choice is automatic — no extra flags needed.

Outputs three files: the thresholded adjacency (`.npy`), a signed network (`_signed.npy`), and a text summary (`.txt`).

Supported input formats: `.csv`, `.tsv`, `.h5ad`, `.npy`.

### NestBoot FDR (Python API)

```python
from sparselink.bench import NestBoot

nestboot = NestBoot(method_name="lasso", n_bootstraps=50, fdr=0.05)
final_adj = nestboot.run(X)
```

### Evaluate a predicted GRN

Compare a predicted adjacency matrix against a gold standard:

```bash
pygs evaluate pred.npy --gold gold.npy
pygs evaluate pred.csv --gold gold_standard.json
```

Prints AUROC, AUPR, F1, MCC, and other metrics. Accepts `.npy`, `.csv`, `.tsv` for predictions and additionally `.json` (GeneSpider format) for gold standards.

### Plot a GRN

Visualize a gene regulatory network:

```bash
pygs plot adjacency.npy --genes genes.txt
pygs plot adjacency.csv --genes genes.txt --tfs tfs.txt --threshold 0.1 -o network.png
```

Gene names default to `G1, G2, ...` if `--genes` is omitted. Optionally highlight transcription factors with `--tfs`.

### Interactive dashboard

After any benchmark run, generate a click-to-drill-down HTML dashboard:

```bash
sparselink-tui dashboard -i benchmark_results.json
```

Drill order: **SNR → Topology → Sparsity**, with methods always compared on the x-axis. Includes collapsible static comparison grids and a searchable data table.

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

All methods implement `InferenceMethod.fit(X) -> InferenceResult`. See [docs/site/methods.md](docs/site/methods.md) for details.

## Project Structure

```
pyGS/
├── src/                  # pyGS source (bio workflows, legacy methods, bootstrap)
│   ├── methods/          # pyGS method wrappers
│   ├── datastruct/       # Network, Dataset classes
│   ├── analyze/          # CompareModels, Data loading
│   ├── bio/              # Biology-specific subpackage
│   └── bootstrap/        # NB-FDR analysis
├── sparselink/           # Standalone inference library
│   └── src/sparselink/
│       ├── methods/      # 20 registered inference methods
│       ├── bench/        # Benchmarking, TUI, dashboard, NestBoot
│       └── accel.py      # MLX acceleration for Apple Silicon
├── benchmark_genespider.py  # GeneSpider dataset benchmark
├── docs/site/            # MkDocs documentation
└── pyproject.toml
```

## Development

```bash
git clone https://github.com/dcolinmorgan/pyGS.git
cd pyGS
pip install -e ".[dev]"
pip install -e "sparselink/[dev]"

# Lint & type-check
ruff check src/ sparselink/
mypy

# Test
pytest --cov
```

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

## License

MIT
