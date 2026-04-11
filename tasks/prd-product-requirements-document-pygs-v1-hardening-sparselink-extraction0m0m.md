
# Product Requirements Document: pyGS v1 Hardening & sparselink Extraction

## 1. Overview

Project: Refactor pyGS into a layered architecture with a standalone general-purpose network inference package (sparselink) and a restructured pyGS split into pyGS.bench (benchmarking/data-creation) and pyGS.bio (biology-specific methods).

Goal: Ship a production-ready v1 with clean separation of concerns, enabling sparselink to serve as a general-purpose dependency usable outside biology, while pyGS remains the domain-specific orchestration layer.

Target Date: TBD

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 2. Architecture

sparselink (standalone PyPI package)
├── correlation/info-theoretic (Partial Correlation, PCMCI, Granger, Transfer Entropy)
├── sparse regression (LASSO, LSCO, Elastic Net, Ridge)
├── graphical models (GLASSO, StARS, Neighborhood Selection)
├── tree-based (GENIE3-style, TIGRESS-style)
├── causal discovery (PC, FCI, NOTEARS, DAG-GNN)
└── ensemble/meta (Bayesian structure learning, BDeu/BGe)

pyGS (refactored, depends on sparselink)
├── pyGS.bench — benchmarking & data creation
│   ├── synthetic data generation
│   ├── evaluation metrics (AUROC, AUPR, FDR)
│   ├── NestBoot orchestration (bootstrap + FDR control)
│   └── pipeline runners / CLI
└── pyGS.bio — biology-specific methods
    ├── GRN-specific wrappers (scenicplus, pyscenic, etc.)
    ├── biology-aware preprocessing (expression matrices, TF lists)
    └── domain-specific visualization


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 3. Package: sparselink

### 3.1 Scope

Separate git repo + PyPI package. Contains only domain-agnostic inference algorithms that recover sparse dependency structures from tabular data.

### 3.2 API Design

- Unified interface: each method implements fit(X) -> AdjacencyMatrix or fit(X, y) -> EdgeWeights
- Common types: AdjacencyMatrix, EdgeList, InferenceResult
- Method registry pattern for discovery/enumeration
- All methods accept numpy arrays / pandas DataFrames — no biology-specific inputs

### 3.3 Algorithms (v1)

| Category | Methods |
|----------|---------|
| Correlation / Info-theoretic | Partial Correlation, PCMCI, Granger Causality, Transfer Entropy |
| Sparse regression | LASSO, LSCO (ported from pyGS), Elastic Net, Ridge |
| Graphical models | Graphical LASSO, GLASSO+StARS, Neighborhood Selection (Meinshausen-Bühlmann) |
| Tree-based | Random Forest importance (GENIE3-style), TIGRESS-style (stability selection + LARS) |
| Causal discovery | PC algorithm, FCI, NOTEARS, DAG-GNN |
| Ensemble / Meta | Bayesian structure learning (BDeu/BGe scoring) |

### 3.4 Non-functional Requirements

- Python ≥ 3.10
- Core deps: numpy, scipy, scikit-learn, pandas
- Optional deps groups: [causal] (causallearn, notears), [deep] (torch for DAG-GNN)
- Fully typed (mypy strict)
- 80%+ test coverage target

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 4. Package: pyGS (refactored)

### 4.1 pyGS.bench

- Synthetic network + expression data generation
- Evaluation metrics (AUROC, AUPR, precision, recall, FDR)
- NestBoot: bootstrap aggregation + FDR control (stays here — it's orchestration, not a core algorithm)
- Pipeline runner: run N methods from sparselink on M datasets, collect results
- CLI entry point for batch benchmarking

### 4.2 pyGS.bio

- Biology-specific preprocessing (expression matrices, TF/target gene lists, regulon formatting)
- Wrappers for external bio tools (scenicplus, pyscenic)
- Domain-specific visualization (network plots with gene annotations)
- GRN-specific evaluation (gold standard comparison)

### 4.3 Dependency

toml
[project]
dependencies = ["sparselink>=1.0.0"]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 5. v1 Hardening (applies to both packages)

### 5.1 Versioning & Release

- Semantic versioning (CalVer or SemVer — start with SemVer 1.0.0)
- CHANGELOG.md (Keep a Changelog format)
- GitHub Actions release workflow: tag → build → publish to PyPI
- Branch protection on main

### 5.2 Code Quality

- Type annotations on all public APIs (mypy strict mode)
- Linting: ruff (replaces flake8 + isort + pyupgrade)
- Formatting: ruff format (black-compatible)
- Pre-commit hooks config

### 5.3 Public API

- Explicit __all__ in every __init__.py
- Deprecation policy: warnings for 1 minor version before removal
- Stable import paths documented

### 5.4 Documentation

- MkDocs + mkdocstrings (Material theme)
- Published to GitHub Pages
- API reference auto-generated from docstrings
- Quickstart guide, method comparison table
- Contributing guide

### 5.5 Testing

- pytest + pytest-cov
- CI matrix: Python 3.10, 3.11, 3.12
- Unit tests for each algorithm
- Integration tests for pipeline runner

### 5.6 Packaging

- pyproject.toml (PEP 621), build backend: hatchling or setuptools
- src/ layout retained
- Optional dependency groups: [dev], [docs], [test]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 6. Migration Plan

| Phase | Deliverable |
|-------|-------------|
| 1 | Create sparselink repo, port LASSO/LSCO/CLR with unified interface |
| 2 | Add remaining algorithm stubs + implementations (tree-based, graphical models) |
| 3 | Add causal discovery + ensemble methods (can depend on optional extras) |
| 4 | Refactor pyGS into pyGS.bench + pyGS.bio, replace inline methods with sparselink imports |
| 5 | Hardening pass: types, linting, docs, CI/CD, changelog |
| 6 | v1.0.0 release of both packages |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 7. Success Criteria

- pip install sparselink works standalone with no biology deps
- pip install pyGS pulls in sparselink automatically
- All existing pyGS functionality preserved (no regression)
- mypy passes strict on both packages
- Docs site live with API reference + quickstart
- GitHub Actions green: lint + test + publish pipeline
- At least 12 inference algorithms available in sparselink v1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


## 8. Open Questions

1. Should sparselink support GPU acceleration for NOTEARS/DAG-GNN in v1, or defer to v1.1?
2. Preferred license for sparselink? (MIT, Apache-2.0, BSD-3?)
3. Do you want a CLI for sparselink itself, or only through pyGS.bench?
4. Any existing tests in pyGS to preserve/migrate?
