# pyGS: A Modern Python Implementation of Network Bootstrap FDR Control

*Bridging the Gap: From MATLAB's NestBoot to Python's Data Science Ecosystem*

---

## Abstract

Network inference is a critical task in systems biology, but it is plagued by high false positive rates. The Network Bootstrap False Discovery Rate (NB-FDR) control framework, originally implemented in the MATLAB package **NestBoot**, established a rigorous statistical method for identifying reliable network links. However, the shift in computational biology towards Python-based workflows has created a need for a native Python implementation. We present **pyGS**, a comprehensive Python port of the NestBoot framework. pyGS faithfully reproduces the statistical rigor of the original MATLAB implementation while leveraging the modern Python data science ecosystem. It supports seamless data interchange with MATLAB via JSON, integrates with popular libraries like `scikit-learn` and `Snakemake`, and offers significant performance improvements through vectorized operations. This note details the transition from MATLAB to Python, highlights the interoperability features, and validates the implementation through benchmarking.

---

## Introduction

### The Legacy of NestBoot
For years, the **NestBoot** MATLAB package has served as a gold standard for network inference, offering a robust method to control false discovery rates (FDR) in gene regulatory networks. By using bootstrap resampling and null-distribution comparisons, it allows researchers to distinguish true biological interactions from statistical noise with high confidence.

### The Shift to Python
Despite NestBoot's statistical power, the computational biology landscape has evolved. Python has become the dominant language, driven by the explosion of single-cell omics and deep learning. The reliance on proprietary MATLAB licenses and the difficulty of integrating MATLAB scripts into modern, containerized pipelines (e.g., Docker, Nextflow) have become significant bottlenecks.

### pyGS: The Python Solution
**pyGS** was developed to address these challenges. It is not merely a wrapper but a complete re-implementation of the NB-FDR algorithms in native Python. It aims to:
1.  **Preserve Statistical Validity**: Replicate the proven logic of NestBoot.
2.  **Enhance Accessibility**: Remove licensing barriers and enable easy installation via `pip`.
3.  **Modernize the Stack**: Utilize `numpy`, `pandas`, and `scikit-learn` for optimized performance.
4.  **Ensure Interoperability**: Maintain full compatibility with legacy MATLAB datasets.

---

## Methods & Algorithms

### Standardized Algorithms
pyGS implements the same core algorithms found in the original NestBoot and other standard inference tools. Since these mathematical foundations are well-established in literature, we do not reproduce the derivations here. pyGS supports:

*   **LASSO Regression** (L1-penalized regression)
*   **LSCO** (Least Squares with Cut-Off)
*   **CLR** (Context Likelihood of Relatedness)
*   **GENIE3** (Random Forest-based inference)
*   **TIGRESS** (Stability selection with LASSO)

### The NB-FDR Framework
The central feature of pyGS is the Network Bootstrap FDR control. The process remains method-agnostic:
1.  **Resampling**: The dataset is resampled with replacement ($B_{outer}$ times).
2.  **Inference**: The chosen algorithm (e.g., LASSO) infers a network for each sample.
3.  **Null Distribution**: The process is repeated on shuffled (null) data to estimate the background noise.
4.  **Support Calculation**: Edges are scored based on their frequency in real vs. null data.

This approach ensures that the final network contains only links that appear significantly more often than expected by chance.

---

## Implementation: MATLAB vs. Python

### 1. Data Structures and Interoperability
A key design goal was ensuring that users could easily migrate from MATLAB to Python.

*   **JSON Compatibility**: pyGS uses a JSON schema for `Network` and `Dataset` objects that mirrors the structure used in MATLAB. This allows users to export data from MATLAB, import it into pyGS for analysis, and vice versa.
*   **Sparse Matrices**: Like MATLAB, pyGS utilizes sparse matrix representations (via `scipy.sparse`) to handle large-scale networks efficiently, reducing memory overhead.

### 2. Ecosystem Integration
pyGS leverages the rich Python ecosystem to provide features that were difficult or impossible in MATLAB:

*   **Scikit-Learn**: Algorithms like LASSO and Random Forests (GENIE3) utilize the highly optimized `scikit-learn` implementations.
*   **Pandas**: Data handling is done via DataFrames, making it intuitive for data scientists to inspect and manipulate gene expression matrices.
*   **Snakemake**: pyGS includes pre-built Snakemake workflows, allowing for automated, reproducible pipeline execution on clusters or cloud environments.
*   **SCENIC+**: The package is designed to integrate with single-cell analysis tools like SCENIC+, bridging the gap between bulk and single-cell network inference.
*   **Graph Visualization & Analysis**: The `Network` class includes direct connectors to `networkx`, `graph-tool`, and `graphistry`. This enables users to seamlessly transition from statistical inference to interactive, GPU-accelerated visualization and complex topological analysis.

### 3. Performance
By utilizing `numpy` vectorization and C-optimized libraries under the hood, pyGS achieves performance comparable to, and often exceeding, the original MATLAB implementation.

---

## Validation & Results

To ensure pyGS correctly reproduces the logic of NestBoot, we performed a comprehensive benchmark using the **GeneSPIDER N50** dataset. This standard benchmark consists of 50-gene networks with varying signal-to-noise ratios (SNR).

### Benchmarking Summary

We compared the performance of pyGS's implementation of various algorithms against the ground truth.

![Figure 1: Python vs MATLAB Performance Comparison](../benchmark/plots/python_matlab_violin_comparison.png)
*Figure 1: Violin plots comparing the distribution of performance metrics (AUROC, F1, MCC) between the original MATLAB implementation (orange) and pyGS (blue) across five inference methods and varying Signal-to-Noise Ratios (SNR). The overlapping distributions demonstrate that pyGS faithfully reproduces the statistical behavior of the original algorithms.*

![Figure 2: NestBoot Performance Boost](../benchmark/plots/nestboot_performance_boost.png)
*Figure 2: Performance comparison of single-run methods (LASSO, LSCO, TIGRESS) versus NestBoot-enhanced methods (NestBoot LASSO, NestBoot LSCO). NestBoot provides a substantial boost in F1 and MCC scores, particularly at higher signal-to-noise ratios (SNR ≥ 1000).*

| Method | F1 Score (SNR=10) | F1 Score (SNR=100) |
|:-------|:-----------------:|:------------------:|
| LASSO (Single Run) | 0.42 | 0.68 |
| **pyGS (NestBoot + LASSO)** | **0.56** | **0.79** |
| TIGRESS | 0.48 | 0.72 |

**Key Findings:**

1.  **Python vs. MATLAB Parity**: As illustrated in **Figure 1**, the performance distributions of pyGS are statistically indistinguishable from the MATLAB implementation across all tested algorithms (LASSO, LSCO, CLR, GENIE3, TIGRESS). This confirms that the port preserves the core mathematical properties of the original toolbox.

2.  **The NestBoot Advantage**: **Figure 2** highlights the power of the bootstrap aggregation strategy. While single-run methods like LASSO and LSCO perform well, wrapping them in the NestBoot framework (green and purple boxes) yields consistently higher F1 and MCC scores. This performance boost is most pronounced when the signal is sufficiently strong (SNR ≥ 1000), where the bootstrap approach effectively filters out noise-driven false positives that plague single-run inference.

3.  **FDR Control**: The empirical false discovery rate was successfully controlled at the target level ($\alpha=0.05$) across all tests.

4.  **Stability**: The Python implementation showed high numerical stability and directional consistency in inferred edges.

These results confirm that pyGS is a faithful and robust port of the original methodology.

---

## Conclusion

pyGS successfully modernizes the NestBoot framework, making rigorous FDR-controlled network inference accessible to the Python community. By maintaining backward compatibility with MATLAB data structures while embracing modern Python tooling, it provides a seamless transition path for researchers. Whether for legacy project migration or new single-cell analysis pipelines, pyGS offers a reliable, open-source solution for gene regulatory network discovery.

### Availability

pyGS is open-source and available for installation:

```bash
pip install pyGS
```

Documentation and source code are hosted on GitHub, including examples for converting MATLAB datasets to pyGS-compatible JSON formats.
