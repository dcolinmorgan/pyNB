# pyGS & sparselink

**Python GENESpider** — gene regulatory network inference and benchmarking.

pyGS provides a unified framework for inferring gene regulatory networks from expression data, with bootstrap-based FDR control and multiple inference algorithms via the **sparselink** engine.

## Features

- **20+ inference methods** — from Lasso to NOTEARS to DAG-GNN
- **Unified API** — `method.fit(X)` returns `InferenceResult` for every algorithm
- **NestBoot FDR control** — bootstrap aggregation with false discovery rate thresholding
- **Benchmarking suite** — synthetic data generation, AUROC/AUPR evaluation
- **Biology integration** — SCENIC+, regulon formatting, gold standard comparison
- **Fully typed** — PEP 561 compliant with strict mypy

## Quick links

- [Quickstart](quickstart.md) — get running in 5 minutes
- [Method Comparison](methods.md) — choose the right algorithm
- [API Reference](api/sparselink.md) — full auto-generated docs
- [Contributing](contributing.md) — how to add methods or fix bugs
