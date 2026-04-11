# Changelog

All notable changes to sparselink will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-04-11

### Added
- Unified `fit(X) -> InferenceResult` interface via `InferenceMethod` base class
- Method registry with `@registry.register` decorator for discovery
- Common types: `AdjacencyMatrix`, `EdgeList`, `InferenceResult`, `InputData`
- 20 inference methods across 5 categories:
  - Regularization: Lasso, Elastic Net, Ridge, LSCO
  - Information-theoretic: CLR, Partial Correlation
  - Causal: PCMCI, Granger Causality, Transfer Entropy
  - Graphical models: Graphical LASSO, GLASSO+StARS, Neighborhood Selection
  - Tree/stability: GENIE3, TIGRESS
  - Constraint-based: PC, FCI
  - Continuous optimization: NOTEARS, DAG-GNN
  - Bayesian: BDeu, BGe
- `sparselink.bench` module: synthetic data, evaluation metrics, NestBoot, pipeline runner, CLI
- Full PEP 561 type annotations (py.typed)
- 94% test coverage with pytest-cov gate at 80%

[Unreleased]: https://github.com/dcolinmorgan/pyGS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dcolinmorgan/pyGS/releases/tag/v1.0.0
