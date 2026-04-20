# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `pygs` CLI with interactive mode and subcommands (`status`, `methods`, `infer`, `bench`, `bench-gs`, `nestboot`, `evaluate`, `plot`, `show`, `dashboard`)
- GeneSpider live benchmark runner with Rich progress bars in the `pygs` CLI
- NestBoot tier option in the interactive `bench-gs` wizard — select "nestboot" alongside method tiers to configure outer/inner runs and FDR threshold inline

### Changed
- Direct baseline in `bench-gs` now averages absolute adjacency matrices across the alpha sweep (bagging over regularization strengths) instead of selecting a single alpha by bootstrap stability

## [1.0.0] - 2026-04-11

### Added
- **sparselink** standalone package for domain-agnostic sparse network inference
- Unified `fit(X) -> InferenceResult` interface for all inference methods
- Method registry with `@registry.register` decorator pattern
- 20 inference methods: Lasso, Elastic Net, Ridge, LSCO, CLR, Partial Correlation,
  PCMCI, Granger Causality, Transfer Entropy, Graphical LASSO, GLASSO+StARS,
  Neighborhood Selection, GENIE3, TIGRESS, PC, FCI, NOTEARS, DAG-GNN, BDeu, BGe
- `sparselink.bench` benchmarking module with synthetic data, metrics, NestBoot, and CLI
- `pyGS.bio` subpackage for biology-specific preprocessing, wrappers, visualization, evaluation
- Full type annotations (PEP 561 py.typed) on all public APIs
- MkDocs documentation site with API reference, quickstart, and method comparison
- GitHub Actions CI/CD (lint, test matrix, PyPI publish, docs deploy)
- pytest-cov with 80% coverage gate on sparselink
- ruff linting and formatting (replaces flake8 + isort + black)
- pre-commit hooks configuration
- Deprecation policy with `DeprecationWarning` utilities

### Changed
- pyGS inference methods now delegate to sparselink (thin wrappers preserve API)
- Packaging migrated to PEP 621 with hatchling build backend
- Biology-heavy dependencies moved to optional extras (`[causal]`, `[deep]`)

### Deprecated
- `pyGS.LegacyNetworkBootstrap` — use `pyGS.NetworkBootstrap` instead
  (will be removed in 1.2.0)

[Unreleased]: https://github.com/dcolinmorgan/pyGS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dcolinmorgan/pyGS/releases/tag/v1.0.0
