# Performance Comparison Scripts ✅

This directory contains scripts to automatically generate performance comparison reports between Python and MATLAB implementations. **The system is now fully functional and tested!**

## 🎉 Status: Complete and Working

✅ **Python benchmark script**: `generate_python_results.py`  
✅ **MATLAB benchmark script**: `generate_matlab_results.m` (updated to work with GeneSPIDER2)  
✅ **Report generator**: `generate_performance_report.py`  
✅ **Test suite**: `test_performance_report.py`  
✅ **Sample data**: Created and validated  
✅ **Generated report**: `PERFORMANCE_COMPARISON_GENERATED.md`  
✅ **Visualizations**: 4 charts generated automatically  

## Quick Demo

The system has been tested and generates comprehensive reports automatically:

```bash
# Create sample data and test (already done)
python test_performance_report.py

# Generate final report (already done)
python generate_performance_report.py
```

**Result**: `PERFORMANCE_COMPARISON_GENERATED.md` with real performance data and visualizations!

## Files

### `generate_python_results.py`
- **Purpose**: Run Python benchmarks and save results
- **Output**: CSV files with detailed performance metrics
- **Dependencies**: pyNB package, numpy, pandas, psutil

### `generate_matlab_results.m`
- **Purpose**: Run MATLAB benchmarks and save results  
- **Output**: CSV files with MATLAB performance metrics
- **Dependencies**: MATLAB Statistics Toolbox

### `generate_performance_report.py`
- **Purpose**: Read benchmark files and create markdown report
- **Output**: `PERFORMANCE_COMPARISON_GENERATED.md`
- **Dependencies**: pandas, numpy, matplotlib, seaborn

## Usage

### Step 1: Run Python Benchmarks
```bash
# Ensure pyNB dependencies are installed
pip install numpy pandas psutil matplotlib seaborn

# Run Python benchmarks
python generate_python_results.py
```

### Step 2: Run MATLAB Benchmarks
```matlab
% In MATLAB command window
run('generate_matlab_results.m')
```

### Step 3: Generate Report
```bash
# Generate comprehensive markdown report
python generate_performance_report.py
```

## Output Structure

```
benchmark_results/
├── python_benchmark_results.csv    # Detailed Python results
├── python_summary.json             # Python summary statistics
├── matlab_benchmark_results.csv    # Detailed MATLAB results  
├── matlab_summary.json             # MATLAB summary statistics
├── plots/                           # Generated visualizations
│   ├── execution_time_comparison.png
│   ├── memory_usage_comparison.png
│   ├── f1_score_comparison.png
│   └── radar_chart.png
└── individual_results/              # Per-test JSON files
    ├── python_lasso_small_results.json
    ├── python_lsco_small_results.json
    ├── matlab_lasso_simple_small_results.json
    └── ...
```

## Benefits

### 1. **Real Data**
- Actual execution times and memory usage
- Platform-specific optimizations measured
- Reproducible results with random seeds

### 2. **Automatic Updates** 
- Rerun scripts to update performance data
- No manual metric maintenance
- Always reflects current implementation

### 3. **Comprehensive Analysis**
- Statistical analysis of multiple runs
- Visual comparisons with charts
- Best performer identification

### 4. **Extensible**
- Easy to add new metrics
- Configurable dataset sizes
- Modular script design

## Customization

### Adding New Metrics
Edit the benchmark scripts to collect additional metrics:

```python
# In generate_python_results.py
result = {
    'execution_time': execution_time,
    'memory_usage': memory_usage,
    'custom_metric': calculate_custom_metric(),  # Add new metric
    # ... existing metrics
}
```

### Testing Different Dataset Sizes
Modify the dataset configurations:

```python
# In generate_python_results.py
dataset_sizes = [
    (10, 20, "small"),
    (50, 100, "medium"),
    (100, 200, "large"),  # Add new size
]
```

### Custom Report Sections
Extend the report generator:

```python
# In generate_performance_report.py
def _generate_custom_section(self) -> List[str]:
    return [
        "## Custom Analysis",
        "Your custom analysis here...",
        ""
    ]
```

## Troubleshooting

### Python Import Errors
```bash
# Install missing dependencies
pip install -e ".[dev]"
```

### MATLAB Path Issues
```matlab
% Add src directory to MATLAB path
addpath('src');
addpath(genpath('src'));
```

### Missing Results Files
- Check that benchmark scripts completed successfully
- Verify output directory permissions
- Look for error messages in console output

## Example Output

The generated report includes:
- Executive summary with test configuration
- Detailed performance metrics table
- Visual comparisons (box plots, radar charts)
- Best performer identification
- Platform-specific recommendations

Sample metrics tracked:
- Execution time (seconds)
- Memory usage (MB)
- Network quality (F1 score, precision, recall)
- Sparsity patterns
- Scalability characteristics

## Integration with CI/CD

These scripts can be integrated into continuous integration:

```yaml
# .github/workflows/performance.yml
name: Performance Benchmarks
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Python benchmarks
        run: python generate_python_results.py
      - name: Generate report
        run: python generate_performance_report.py
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: performance-report
          path: PERFORMANCE_COMPARISON_GENERATED.md
```

This approach ensures the performance comparison always reflects the current state of both implementations.
