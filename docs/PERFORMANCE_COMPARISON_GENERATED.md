# 📊 Performance Comparison Report

**Generated:** 2025-07-18 11:38:24

## Executive Summary

This report compares the performance characteristics of Python pyNB package versus MATLAB implementation for network inference and bootstrap FDR analysis. Results are based on actual benchmark runs with real performance measurements.

---

## 🎯 Test Configuration

- **Total Tests:** 8
- **Platforms:** Python, MATLAB
- **Methods:** LASSO, LSCO, lasso, lsco
- **Dataset Sizes:** small, medium

## 📈 Detailed Performance Analysis

### Performance Metrics Summary

| Platform | Method | Avg Time (s) | Avg Memory (MB) | Avg F1 Score | Avg Precision | Avg Recall |
|----------|--------|-------------|----------------|-------------|--------------|-----------|
| MATLAB | lasso | 19.63 | 14.4 | 0.000 | 1.000 | 0.000 |
| MATLAB | lsco | 0.88 | 16.9 | 0.000 | 1.000 | 0.000 |
| Python | LASSO | 0.40 | 0.2 | 0.077 | 0.114 | 0.079 |
| Python | LSCO | 0.01 | 0.0 | 0.304 | 0.180 | 1.000 |


## 📊 Performance Visualizations

The following charts provide visual comparisons of key performance metrics:

### Execution Time Comparison

![Execution Time Comparison](plots/execution_time_comparison.png)

### F1 Score Comparison

![F1 Score Comparison](plots/f1_score_comparison.png)

### Memory Usage Comparison

![Memory Usage Comparison](plots/memory_usage_comparison.png)

### Radar Chart

![Radar Chart](plots/radar_chart.png)

## 🏆 Best Performers

| Metric | Winner | Platform | Method | Value |
|--------|--------|----------|--------|-------|
| Execution Time | 🥇 | Python | LSCO | 0.000 |
| Memory Usage | 🥇 | MATLAB | lsco | -6.667 |
| F1 Score | 🥇 | Python | LSCO | 0.333 |
| Precision | 🥇 | MATLAB | lasso | 1.000 |
| Sensitivity | 🥇 | Python | LSCO | 1.000 |


## 🎯 Recommendations

Based on the benchmark results:

- **Fastest Execution:** Python LSCO (0.00 seconds)
- **Most Memory Efficient:** MATLAB lsco (-6.7 MB)
- **Best Network Quality:** Python LSCO (F1 Score: 0.333)


---

*Report generated automatically from benchmark data on 2025-07-18*

**Data Sources:**
- Python results: `benchmark_results/python_benchmark_results.csv`
- MATLAB results: `benchmark_results/matlab_benchmark_results.csv`

**Methodology:**
- All tests run with identical synthetic datasets
- Memory usage measured using platform-specific tools
- Execution time measured wall-clock time
- Network quality metrics calculated using standard confusion matrix methods