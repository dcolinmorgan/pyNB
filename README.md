# Network Bootstrap FDR & mini GeneSPIDER

A Python implementation of NB-FDR (Network Bootstrap False Discovery Rate) analysis for network inference. This package implements an algorithm to estimate bootstrap support for network links by comparing measured networks against a shuffled (null) dataset. It computes key metrics such as assignment fractions, evaluates overlap between inferred links, and determines a bootstrap support cutoff at the desired false discovery rate.

## Overview

In high-throughput network analysis, bootstrapping is used to assess the stability of inferred links. NB-FDR leverages bootstrap iterations to compute the assignment fraction (i.e. the frequency at which a link is inferred) and compares these results against a null distribution obtained from shuffled data. The differences between the measured and shuffled data inform the support level guaranteed for a target FDR level.

Key features of this package include:
- **Computation of Assignment Fractions:** For both measured and null networks based on bootstrap runs.
- **Comparison Between Measured and Null Distributions:** To determine a support metric that approximates (1 - FDR).
- **Export of Results:** Summary statistics are saved as a text file.
- **Visualization:** A dual-axis plot displays the bootstrap support metric (left y-axis) and normalized link frequencies (right y-axis) for both normal and shuffled data.
- **Modular Design:** Clear separation of source code, tests, examples, and configuration.
- **Snakemake Workflow:** Automated analysis pipeline for processing multiple samples.
- **SCENIC+ Integration:** Optional integration with scenicplus for comprehensive gene regulatory network analysis.
- **JSON Data Import:** Support for downloading and importing network data from JSON URLs.

## Data and Network Import

The package supports downloading data and networks directly from JSON URLs. This is particularly useful for accessing pre-computed datasets and reference networks.

### Example Usage

```python
from analyze.Data import Data
from datastruct.Network import Network

# Download dataset from JSON URL
dataset = Data.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json'
)

# Download reference network from JSON URL
true_net = Network.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json'

lasso_net,alpha=Lasso(dataset.data)
lasso_net=Network(lasso_net)

comp = CompareModels(lasso_net, true_net)
print(f"lasso F1 Score: {comp.F1}")
[0.32806324]
print(f"lasso MCC : {comp.MCC}")
[0.28972279]

```

## Analysis Output as Figure

<<<<<<< Updated upstream
![Analysis Output](output/analysis_plot.png)
=======
```python
from methods.lsco import LSCO
from methods.lasso import Lasso

print("\n🧪 Network Inference Comparison")
print("="*50)

# Method 1: LASSO Regression (Sparse)
print("🎯 Running LASSO inference...")
lasso_net, alpha = Lasso(dataset.data)
lasso_net = Network(lasso_net)

print(f"   ✅ LASSO completed (α={alpha:.6f})")
print(f"   🕸️ Inferred edges: {np.sum(lasso_net.A != 0)}")
print(f"   📊 Sparsity: {np.sum(lasso_net.A == 0) / lasso_net.A.size:.3f}")

# Method 2: Least Squares (Dense)  
print("\n🎯 Running Least Squares inference...")
lsco_net, mse = LSCO(dataset.data)
lsco_net = Network(lsco_net)

print(f"   ✅ LSCO completed (MSE={mse:.6f})")
print(f"   🕸️ Non-zero elements: {np.sum(np.abs(lsco_net.A) > 1e-6)}")

# Compare both methods against ground truth
print("\n📊 Network Comparison Results")
print("-" * 30)

# LASSO vs True Network
lasso_comp = CompareModels(true_net, lasso_net)
print(f"🎯 LASSO Performance:")
print(f"   F1 Score: {lasso_comp.F1[0]:.4f}")
print(f"   MCC: {lasso_comp.MCC[0]:.4f}")
print(f"   Sensitivity: {lasso_comp.sen[0]:.4f}")
print(f"   Specificity: {lasso_comp.spe[0]:.4f}")
print(f"   Precision: {lasso_comp.pre[0]:.4f}")

# LSCO vs True Network  
lsco_comp = CompareModels(true_net, lsco_net)
print(f"\n📐 LSCO Performance:")
print(f"   F1 Score: {lsco_comp.F1[0]:.4f}")
print(f"   MCC: {lsco_comp.MCC[0]:.4f}")
print(f"   Sensitivity: {lsco_comp.sen[0]:.4f}")
print(f"   Specificity: {lsco_comp.spe[0]:.4f}")
print(f"   Precision: {lsco_comp.pre[0]:.4f}")
```

### 3. 🧮 Bootstrap FDR Analysis (Core Functionality)

```python
from bootstrap.nb_fdr import NetworkBootstrap
import pandas as pd
from pathlib import Path
import numpy as np

print("\n🔬 Network Bootstrap FDR Analysis")
print("="*40)

# Create synthetic bootstrap data for demonstration
def create_bootstrap_data(n_genes=20, n_runs=65, n_links_per_run=50):
    """Generate synthetic network data in the required format."""
    np.random.seed(42)
    data = []
    
    gene_names = [f"Gene_{i:02d}" for i in range(n_genes)]
    
    for run in range(n_runs):
        # Generate random gene pairs
        for _ in range(n_links_per_run):
            gene_i = np.random.choice(gene_names)
            gene_j = np.random.choice(gene_names)
            if gene_i != gene_j:  # Avoid self-loops
                # Normal data: some structure + noise
                base_strength = 0.5 if hash(f"{gene_i}_{gene_j}") % 3 == 0 else 0.0
                link_value = base_strength + np.random.normal(0, 0.3)
                
                data.append({
                    'gene_i': gene_i,
                    'gene_j': gene_j,
                    'run': run,
                    'link_value': link_value
                })
    
    return pd.DataFrame(data)

# Generate normal and shuffled data
print("📊 Generating synthetic bootstrap data...")
normal_data = create_bootstrap_data(n_genes=15, n_runs=65, n_links_per_run=30)
print(f"   ✅ Normal data: {len(normal_data)} entries")

# Shuffled data (null hypothesis)
shuffled_data = normal_data.copy()
shuffled_data['link_value'] = np.random.normal(0, 0.2, len(shuffled_data))
print(f"   ✅ Shuffled data: {len(shuffled_data)} entries")

# Initialize NetworkBootstrap analyzer
nb = NetworkBootstrap()

# Perform comprehensive FDR analysis
print("\n🧮 Running NB-FDR Analysis...")
results = nb.nb_fdr(
    normal_df=normal_data,
    shuffled_df=shuffled_data,
    init=64,                    # Number of bootstrap iterations
    data_dir=Path("output"),    # Output directory
    fdr=0.05,                  # False Discovery Rate threshold
    boot=8                     # Bootstrap group size
)

print("✅ NB-FDR Analysis Complete!")
print(f"   🎯 Support threshold: {results.support:.3f}")
print(f"   📊 False positive rate: {results.fp_rate:.4f}")
print(f"   🕸️ Network edges above threshold: {np.sum(results.xnet)}")
print(f"   📈 Network sparsity: {np.mean(results.xnet == 0):.3f}")

# Export detailed results
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

nb.export_results(results, output_dir / "fdr_analysis_results.txt")
print(f"\n📄 Results exported to: {output_dir / 'fdr_analysis_results.txt'}")

# Create publication-ready visualization
print("\n📈 Generating analysis plots...")

# Prepare data for plotting
agg_normal = nb.compute_assign_frac(normal_data, 64, 8)
agg_normal.rename(columns={'Afrac': 'Afrac_norm', 'Asign_frac': 'Asign_frac_norm'}, inplace=True)

agg_shuffled = nb.compute_assign_frac(shuffled_data, 64, 8)  
agg_shuffled.rename(columns={'Afrac': 'Afrac_shuf', 'Asign_frac': 'Asign_frac_shuf'}, inplace=True)

merged = pd.merge(agg_normal, agg_shuffled, on=['gene_i', 'gene_j'], how='outer').fillna(0)

# Generate bootstrap support plot
nb.plot_analysis_results(merged, output_dir / "bootstrap_analysis.png", bins=15)
print(f"   📊 Bootstrap plot saved: {output_dir / 'bootstrap_analysis.png'}")

# Compute network density metrics
density_results = nb.compute_network_density(normal_data, threshold=0.1)
print(f"\n📊 Network Density Analysis:")
print(f"   📈 Mean density: {density_results['density'].mean():.4f}")
print(f"   📊 Density std: {density_results['density'].std():.4f}")
print(f"   🔗 Mean links per run: {density_results['num_links'].mean():.1f}")
```

### 4. 📊 Advanced Network Analysis and Comparison

```python
# Detailed network structure analysis
print("\n🔬 Advanced Network Analysis")
print("="*35)

# Analyze assignment fractions in detail
print("📈 Assignment Fraction Analysis:")
afrac_stats = agg_normal['Afrac_norm'].describe()
print(f"   Mean assignment fraction: {afrac_stats['mean']:.3f}")
print(f"   Std assignment fraction: {afrac_stats['std']:.3f}")
print(f"   Links with full support (Afrac=1): {sum(agg_normal['Afrac_norm'] >= 1)}")
print(f"   Links with high support (Afrac≥0.8): {sum(agg_normal['Afrac_norm'] >= 0.8)}")

# Sign consistency analysis
sign_stats = agg_normal['Asign_frac_norm'].describe()
print(f"\n🎯 Sign Consistency Analysis:")
print(f"   Mean sign fraction: {sign_stats['mean']:.3f}")
print(f"   Positive regulation bias: {sum(agg_normal['Asign_frac_norm'] > 0.5)}")
print(f"   Negative regulation bias: {sum(agg_normal['Asign_frac_norm'] < -0.5)}")

# Network topology comparison
print(f"\n🕸️ Network Topology Comparison:")
print(f"   True network edges: {np.sum(true_net.A != 0)}")
print(f"   LASSO inferred edges: {np.sum(lasso_net.A != 0)}")
print(f"   Bootstrap significant edges: {np.sum(results.xnet)}")
print(f"   Overlap (True ∩ LASSO): {np.sum((true_net.A != 0) & (lasso_net.A != 0))}")

# Performance summary table
print(f"\n📋 Method Performance Summary:")
print(f"{'Method':<10} {'F1':<8} {'MCC':<8} {'Precision':<10} {'Recall':<8}")
print("-" * 45)
print(f"{'LASSO':<10} {lasso_comp.F1[0]:<8.3f} {lasso_comp.MCC[0]:<8.3f} {lasso_comp.pre[0]:<10.3f} {lasso_comp.sen[0]:<8.3f}")
print(f"{'LSCO':<10} {lsco_comp.F1[0]:<8.3f} {lsco_comp.MCC[0]:<8.3f} {lsco_comp.pre[0]:<10.3f} {lsco_comp.sen[0]:<8.3f}")
```

### 5. 📁 Working with Custom Data

```python
# Example: Creating and analyzing custom network data
print("\n📁 Custom Data Analysis Example")
print("="*32)

# Create custom dataset
custom_Y = np.random.randn(10, 20)  # 10 genes, 20 samples
custom_P = np.random.randn(10, 20)  # Perturbation matrix (same dimensions as Y)

from datastruct.Dataset import Dataset
custom_dataset = Dataset()
custom_dataset._Y = custom_Y
custom_dataset._P = custom_P
custom_dataset._names = [f"CustomGene_{i}" for i in range(10)]

print("✅ Custom dataset created")
print(f"   📊 Shape: {custom_dataset.Y.shape}")
print(f"   🧬 Genes: {custom_dataset.N}")

# Infer network from custom data using both methods
print(f"\n🎯 Custom Network Inference Comparison:")
print("="*45)

# Method 1: LASSO (Sparse)
custom_lasso, custom_alpha = Lasso(custom_dataset)
custom_lasso_network = Network(custom_lasso)

print(f"🎯 LASSO Results:")
print(f"   α parameter: {custom_alpha:.6f}")
print(f"   🕸️ Edges: {np.sum(custom_lasso_network.A != 0)}")
print(f"   📈 Density: {np.sum(custom_lasso_network.A != 0) / custom_lasso_network.A.size:.3f}")
print(f"   📊 Sparsity: {np.sum(custom_lasso_network.A == 0) / custom_lasso_network.A.size:.3f}")

# Method 2: LSCO (Dense)
from methods.lsco import LSCO
custom_lsco, custom_mse = LSCO(custom_dataset)
custom_lsco_network = Network(custom_lsco)

print(f"\n📐 LSCO Results:")
print(f"   MSE: {custom_mse:.6f}")
print(f"   🕸️ Non-zero elements: {np.sum(np.abs(custom_lsco_network.A) > 1e-6)}")
print(f"   📈 Density: {np.sum(np.abs(custom_lsco_network.A) > 1e-6) / custom_lsco_network.A.size:.3f}")
print(f"   📊 Sparsity: {np.sum(np.abs(custom_lsco_network.A) <= 1e-6) / custom_lsco_network.A.size:.3f}")

# Compare sparsity patterns
print(f"\n📊 Method Comparison:")
print(f"   LASSO vs LSCO sparsity ratio: {(np.sum(custom_lasso_network.A == 0) / custom_lasso_network.A.size) / (np.sum(np.abs(custom_lsco_network.A) <= 1e-6) / custom_lsco_network.A.size):.2f}")
print(f"   LASSO edges: {np.sum(custom_lasso_network.A != 0)}")
print(f"   LSCO edges: {np.sum(np.abs(custom_lsco_network.A) > 1e-6)}")

# Network density over different thresholds for both methods
thresholds = [0.0, 0.1, 0.2, 0.5, 1.0]
print(f"\n📊 Network Density vs Threshold Comparison:")
print(f"{'Threshold':<10} {'LASSO Edges':<12} {'LSCO Edges':<12} {'LASSO Density':<14} {'LSCO Density':<14}")
print("-" * 70)
for thresh in thresholds:
    lasso_edges = np.sum(np.abs(custom_lasso_network.A) > thresh)
    lsco_edges = np.sum(np.abs(custom_lsco_network.A) > thresh)
    lasso_density = lasso_edges / custom_lasso_network.A.size
    lsco_density = lsco_edges / custom_lsco_network.A.size
    print(f"{thresh:<10.1f} {lasso_edges:<12d} {lsco_edges:<12d} {lasso_density:<14.3f} {lsco_density:<14.3f}")
```

## 🔬 MATLAB Integration Example

For users who prefer MATLAB, here's how to replicate the custom data analysis workflow:

```matlab
% GeneSPIDER2 MATLAB Workflow
% Implements pyNB 5-step workflow with non-NestBoot LASSO and LSCO runs
% Dataset: Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json
% Network: Tjarnberg-D20150910-random-N50-L158-ID252384.json
% 8 inner and 8 outer bootstrap runs for NestBoot, seed 42
% Saves outputs as CSV
% Date: 2025-07-18

%% Setup
clear all;
rng(42); % Set random seed
output_dir = 'matlab_output';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
addpath('/path/to/GeneSPIDER2'); % Add GeneSPIDER2 library to path

%% Step 1: Network Data Import and Analysis
fprintf('\n📥 Step 1: Network Data Import and Analysis\n');
fprintf('=====================================\n');

% Load dataset
dataset_url = 'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json';
data = datastruct.Dataset.fetch(dataset_url);
fprintf('✅ Dataset loaded: %s\n', data.dataset);
fprintf('   📊 Expression matrix shape: [%d, %d]\n', size(data.Y));
fprintf('   🧬 Number of genes: %d\n', data.N);
fprintf('   🔬 Number of samples: %d\n', data.M);

% Load reference network
network_url = 'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json';
net = datastruct.Network.fetch(network_url);
fprintf('✅ Reference network loaded: %s\n', net.network);

%% Step 2: Network Inference
fprintf('\n🔍 Step 2: Network Inference\n');
fprintf('=====================================\n');

zetavec = logspace(-6, 0, 30); % Sparsity parameters
FDR = 0.05; % False Discovery Rate
nest = 8; % Outer bootstrap runs
boot = 8; % Inner bootstrap iterations
par = true; % Parallel processing
cpus = 2; % Number of CPUs

% Non-NestBoot LASSO
fprintf('Running non-NestBoot LASSO...\n');
estA_lasso = Methods.lasso(data, net, zetavec, false);
% Non-NestBoot LSCO
fprintf('Running non-NestBoot LSCO...\n');
estA_lsco = Methods.lsco(data, net, zetavec, false, 'input');

% NestBoot LASSO
fprintf('Running NestBoot LASSO...\n');
nbout_lasso = Methods.NestBoot(data, 'lasso', nest, boot, zetavec, FDR, output_dir, par, cpus);
% NestBoot LSCO
fprintf('Running NestBoot LSCO...\n');
nbout_lsco = Methods.NestBoot(data, 'lsco', nest, boot, zetavec, FDR, output_dir, par, cpus);

%% Step 3: Model Comparison
fprintf('\n📊 Step 3: Model Comparison\n');
fprintf('=====================================\n');

% Compare non-NestBoot LASSO
M_lasso = analyse.CompareModels(net, estA_lasso);
results_lasso = struct(M_lasso);
fprintf('Non-NestBoot LASSO Results:\n');
fprintf('F1 Score: %.3f\n', results_lasso.F1(end));
fprintf('MCC: %.3f\n', results_lasso.MCC(end));
fprintf('AUROC: %.3f\n', M_lasso.AUROC());

% Compare non-NestBoot LSCO
M_lsco = analyse.CompareModels(net, estA_lsco);
results_lsco = struct(M_lsco);
fprintf('Non-NestBoot LSCO Results:\n');
fprintf('F1 Score: %.3f\n', results_lsco.F1(end));
fprintf('MCC: %.3f\n', results_lsco.MCC(end));
fprintf('AUROC: %.3f\n', M_lsco.AUROC());

% Compare NestBoot LASSO
M_nest_lasso = analyse.CompareModels(net, nbout_lasso.binary_networks);
results_nest_lasso = struct(M_nest_lasso);
fprintf('NestBoot LASSO Results:\n');
fprintf('F1 Score: %.3f\n', results_nest_lasso.F1(end));
fprintf('MCC: %.3f\n', results_nest_lasso.MCC(end));
fprintf('AUROC: %.3f\n', M_nest_lasso.AUROC());

% Compare NestBoot LSCO
M_nest_lsco = analyse.CompareModels(net, nbout_lsco.binary_networks);
results_nest_lsco = struct(M_nest_lsco);
fprintf('NestBoot LSCO Results:\n');
fprintf('F1 Score: %.3f\n', results_nest_lsco.F1(end));
fprintf('MCC: %.3f\n', results_nest_lsco.MCC(end));
fprintf('AUROC: %.3f\n', M_nest_lsco.AUROC());

%% Step 4: Save Results
fprintf('\n💾 Step 4: Save Results\n');
fprintf('=====================================\n');

% Save non-NestBoot results as CSV
save_path_lasso = fullfile(output_dir, 'comparison_results_lasso.csv');
M_lasso.save(save_path_lasso, 'csv');
save_path_lsco = fullfile(output_dir, 'comparison_results_lsco.csv');
M_lsco.save(save_path_lsco, 'csv');

% Save NestBoot results as CSV
save_path_nest_lasso = fullfile(output_dir, 'comparison_results_nest_lasso.csv');
M_nest_lasso.save(save_path_nest_lasso, 'csv');
save_path_nest_lsco = fullfile(output_dir, 'comparison_results_nest_lsco.csv');
M_nest_lsco.save(save_path_nest_lsco, 'csv');

% Save inferred networks as CSV
save_network_lasso = fullfile(output_dir, 'inferred_network_lasso.csv');
writematrix(estA_lasso(:, :, end), save_network_lasso);
save_network_lsco = fullfile(output_dir, 'inferred_network_lsco.csv');
writematrix(estA_lsco(:, :, end), save_network_lsco);
save_network_nest_lasso = fullfile(output_dir, 'inferred_network_nest_lasso.csv');
writematrix(nbout_lasso.binary_networks(:, :, end), save_network_nest_lasso);
save_network_nest_lsco = fullfile(output_dir, 'inferred_network_nest_lsco.csv');
writematrix(nbout_lsco.binary_networks(:, :, end), save_network_nest_lsco);

%% Step 5: Visualize Results
fprintf('\n📈 Step 5: Visualize Results\n');
fprintf('=====================================\n');

% Plot ROC curves
figure;
subplot(2, 2, 1);
M_lasso.ROC();
title('Non-NestBoot LASSO ROC Curve');
subplot(2, 2, 2);
M_lsco.ROC();
title('Non-NestBoot LSCO ROC Curve');
subplot(2, 2, 3);
M_nest_lasso.ROC();
title('NestBoot LASSO ROC Curve');
subplot(2, 2, 4);
M_nest_lsco.ROC();
title('NestBoot LSCO ROC Curve');

% Display true network
net.show();
```

### Key Features of the MATLAB Version:

1. **🔧 Data Creation**: Generates synthetic expression and perturbation matrices
2. **🎯 LASSO Regression**: Uses MATLAB's built-in `lasso()` function with cross-validation
3. **📊 Network Analysis**: Calculates density, sparsity, and edge statistics
4. **💾 Data Export**: Saves results in CSV format for Python integration
5. **🧮 Bootstrap Simulation**: Generates multiple bootstrap runs for FDR analysis
6. **📈 Visualization**: Creates comprehensive plots of the analysis results

### Python Integration:

The MATLAB script exports data that can be directly used with the Python pyNB package:

```python
import pandas as pd
import numpy as np
from bootstrap.nb_fdr import NetworkBootstrap

# Load MATLAB-generated data
normal_data = pd.read_csv('matlab_output/bootstrap_data.csv')
shuffled_data = normal_data.copy()
shuffled_data['link_value'] = np.random.normal(0, 0.2, len(shuffled_data))

# Continue with NB-FDR analysis
nb = NetworkBootstrap()
results = nb.nb_fdr(normal_data, shuffled_data, init=64, fdr=0.05)
```

## 🏗️ Package Architecture
>>>>>>> Stashed changes

## Package Structure
```
pyNB/
├── pyproject.toml         # Build and dependency configuration
├── README.md              # Package overview and usage guide
├── src/
│   └── analyze/
│   │   ├── CompareModels.py
│   │   ├── Data.py
│   │   ├── DataModel.py
│   │   ├── Model.py
│   └── bootstrap/
│   │   ├── __init__.py
│   │   ├── nb_fdr.py      # Core implementation of NB-FDR analysis
│   │   ├── utils.py       # Utility functions for network analysis
│   │   └── workflow/      # Snakemake workflow for automated analysis
│   │       ├── __init__.py
│   │       ├── Snakefile
│   │       ├── config/
│   │       │   └── config.yaml
│   │       └── scripts/
│   │           ├── compute_assign_frac.py
│   │           ├── nb_fdr_analysis.py
│   │           ├── generate_plots.py
│   │           └── compute_density.py
│   └── datastruct/
│   │   ├── Dataset.py
│   │   ├── Exchange.py
│   │   ├── Experiment.py
│   │   ├── Network.py
│   └── methods/
│       ├── lasso.py
│       ├── lsco.py
├── tests/
│   ├── __init__.py
│   └── test_py   # Pytest-based tests
└── examples/
    ├── basic_usage.py     # Example script demonstrating package usage
    └── run_workflow.py    # Example script for running the workflow
```

## Installation

The recommended way to install the package is to use a virtual environment. For example:

```bash
python -m venv venv
source venv/bin/activate           # On Windows use: venv\Scripts\activate
pip install -e ".[dev]"            # For GeneSPIDER partial-functionality
pip install -e ".[workflow]"       # For Snakemake workflow and SCENIC+ capabilities
```

This installs all required dependencies including `numpy`, `pandas`, `matplotlib`, and `pytest`. If you install with the `workflow` extra, you'll also get `snakemake` and `scenicplus` for running the automated analysis pipeline and gene regulatory network analysis.

## Usage

### Basic API Usage

A complete working example is provided in the `examples/basic_usage.py` file. In summary, the workflow is as follows:

1. **Process Input Data:**  
   Load CSV files containing network data. Each file should include columns `gene_i`, `gene_j`, `run`, and `link_value` where `run` indicates the bootstrap run number.

2. **Compute Assignment Fractions:**  
   Use the `compute_assign_frac()` method to calculate the frequency (Afrac) and sign fraction (Asign_frac) for each network link.

3. **Merge Measured and Null Data:**  
   Combine the calculated metrics for the normal and shuffled networks.

4. **Run NB-FDR Analysis:**  
   Call the `nb_fdr()` method to compute core network metrics, which returns a `NetworkResults` dataclass.

5. **Export and Visualize Results:**  
   - **Text Summary:** Use `export_results()` to generate a text file summary.
   - **Visualization:** Use `plot_analysis_results()` to create a dual-axis plot. The left y-axis displays a support metric (calculated as the difference in link frequencies between measured and null data normalized by the measured frequency, approximating (1 - FDR)), while the right y-axis shows normalized link frequency distributions.

Example:

```python
from pathlib import Path
from nb_fdr import NetworkBootstrap
import pandas as pd

def process_network_data(data_path: str, is_null: bool = False) -> pd.DataFrame:
    """Process raw network data from a CSV file."""
    df = pd.read_csv(data_path)
    df['run'] = df.run.str.extract(r'(\d+)').astype(int)
    return df[df['run'] < 65].sort_values('run')

def main() -> None:
    """Main execution function."""
    # Load data
    normal_data = process_network_data('../data/normal_data.gz')
    null_data = process_network_data('../data/null_data.gz', is_null=True)
    
    # Initialize analyzer
    nb = NetworkBootstrap()
    
    # Run NB-FDR analysis
    results = nb.nb_fdr(
        normal_df=normal_data,
        shuffled_df=null_data,
        init=64,
        data_dir=Path("output"),
        fdr=0.05,
        boot=8
    )
    
    # Print key results
    print(f"Network sparsity: {(results.xnet != 0).mean():.3f}")
    print(f"Node count: {results.xnet.shape[0]:.3f}")
    print(f"Edge count: {results.xnet.sum():.3f}")
    print(f"False positive rate: {results.fp_rate:.3f}")
    print(f"Support threshold: {results.support:.3f}")

    # Export results and plot analysis
    nb.export_results(results, Path("output/results.txt"))
    
    # Re-create merged DataFrame for plotting
    agg_normal = nb.compute_assign_frac(normal_data, 64, 8)
    agg_normal.rename(columns={'Afrac': 'Afrac_norm', 'Asign_frac': 'Asign_frac_norm'}, inplace=True)
    agg_shuffled = nb.compute_assign_frac(null_data, 64, 8)
    agg_shuffled.rename(columns={'Afrac': 'Afrac_shuf', 'Asign_frac': 'Asign_frac_shuf'}, inplace=True)
    merged = pd.merge(agg_normal, agg_shuffled, on=['gene_i', 'gene_j'])
    
    nb.plot_analysis_results(merged, Path("output/analysis_plot.png"), bins=32)

if __name__ == '__main__':
    main()
```

### Using the Snakemake Workflow

The package includes a Snakemake workflow for automating analysis of multiple samples. To use it:

1. **Create Workflow Directory:**

```python
from pyNB import create_workflow_directory

# Create a directory with Snakefile and config.yaml
workflow_dir = create_workflow_directory("my_workflow", overwrite=True)
```

2. **Prepare Input Data:**

Organize your input data in the format expected by the workflow:
- Place normal data files at: `<output_dir>/data/<sample>/normal_data.csv`
- Place shuffled data files at: `<output_dir>/data/<sample>/shuffled_data.csv`

3. **Edit Configuration:**

Modify the `config/config.yaml` file to specify samples and parameters.

4. **Run the Workflow:**

```python
from pyNB import run_workflow

# Dry run to check that everything is set up correctly
run_workflow("my_workflow", dry_run=True)

# Actual run with 4 cores
run_workflow("my_workflow", cores=4)
```

Alternatively, you can run the workflow directly with the `snakemake` command:

```bash
cd my_workflow
snakemake --cores 4
```

5. **Examine Results:**

The workflow generates:
- Assignment fraction data in `<output_dir>/processed/<sample>/`
- Analysis results in `<output_dir>/results/<sample>/`
- Plots in `<output_dir>/plots/<sample>/`
- Network density information in `<output_dir>/density/`

### Integration with SCENIC+

The package can be used in conjunction with SCENIC+ for comprehensive gene regulatory network analysis. When you install the package with the `workflow` extra dependencies, you'll have access to SCENIC+ functionality that can be used to:

1. Run network inference using SCENIC+ methods
2. Evaluate networks with bootstrapped FDR through our NB-FDR implementation
3. Visualize and analyze results within a unified framework

To use SCENIC+ with NB-FDR:

1. **Install the package with workflow dependencies:**
   ```bash
   pip install -e ".[workflow]"
   ```

2. **Create a custom Snakefile that combines SCENIC+ and NB-FDR:**
   You can adapt the example Snakefile in `src/pyNB/workflow/Snakefile` and the SCENIC+ Snakefile to create a workflow that:
   - Runs SCENIC+ to infer networks
   - Uses bootstrapping for multiple iterations
   - Runs NB-FDR to assess stability and significance
   - Produces integrated reports and visualizations

3. **Recommended directory structure for SCENIC+ integration:**
   ```
   project/
   ├── config/
   │   └── config.yaml       # Combined configuration
   ├── data/
   │   ├── reference/        # Reference files for SCENIC+
   │   └── input/            # Input files
   ├── results/
   │   ├── scenic/           # SCENIC+ results
   │   └── nb_fdr/           # NB-FDR results
   └── Snakefile             # Combined workflow file
   ```


## Limited GeneSpider Functionality

This package includes a limited implementation of network inference methods inspired by GeneSpider. Currently supported methods include:

### Network Inference Methods

1. **LASSO Regression** (`methods.lasso.Lasso`):
   - Solves Y = A^-1*P - E using L1-regularized regression
   - Features:
     - Cross-validation for optimal regularization parameter
     - Sparse network solutions
     - Handles both single and list lambda values
   - Usage:
     ```python
     from methods.lasso import Lasso
     A, alpha = Lasso(dataset, alpha_range=None, cv=5)
     ```

2. **Least Squares** (`methods.lsco.LSCO`):
   - Solves Y = A^-1*P - E using ordinary least squares
   - Features:
     - Non-sparse network solutions
     - Computes mean squared error
     - Handles singular value decomposition
   - Usage:
     ```python
     from methods.lsco import LSCO
     A, mse = LSCO(dataset, tol=1e-8)
     ```

### Data Requirements

Both methods require a Dataset object with:
- `Y`: Expression data matrix (n_genes × n_samples)
- `P`: Perturbation matrix (n_genes × n_samples)

The matrices must have matching dimensions (same number of rows).

### Limitations

This is a limited implementation and does not include:
- Full GeneSpider workflow
- Advanced network inference algorithms
- Integration with other network analysis tools
- Additional preprocessing steps
- Advanced visualization capabilities

For full GeneSpider functionality, please refer to the original implementation.

```n.b. this package is not meant to run network inference, only to compute the FDR based on the inferred networks from multiple bootstrap runs. However, installing [workflow] installs tools needed to repeat figure below (i.e. snakemake & scenic+) ```


## Testing

To run the tests with pytest, simply execute:

```bash
pytest
```

This command will run all tests contained in the `tests/` directory.

## Contributing

Contributions and feedback are welcome! Please open issues or submit pull requests on GitHub.

## References

- [CancerGRN Analysis Example](https://dcolin.shinyapps.io/cancergrn/)
- [Bioinformatics Article](https://academic.oup.com/bioinformatics/article/35/6/1026/5086392)
- [SCENIC+ Documentation](https://scenicplus.readthedocs.io/)

## License

This project is licensed under the [Your License Name] License.
