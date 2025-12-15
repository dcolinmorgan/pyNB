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

# Network Bootstrap FDR & mini GeneSPIDER

A comprehensive Python implementation of NB-FDR (Network Bootstrap False Discovery Rate) analysis for gene regulatory network inference and evaluation. This package implements an algorithm to estimate bootstrap support for network links by comparing measured networks against a shuffled (null) dataset, providing robust statistical assessment of network inference results.

## 🚀 Key Features

- **🧮 Network Bootstrap FDR Analysis:** Core implementation for assessing network link reliability
- **🕸️ Network Inference Methods:** LASSO and Least Squares approaches for network reconstruction  
- **📊 Data Import/Export:** JSON URL support for datasets and reference networks
- **📈 Network Comparison:** Comprehensive metrics (F1, MCC, sensitivity, specificity)
- **📉 Visualization:** Publication-ready plots for bootstrap analysis results
- **🔬 Network Analysis:** Density calculations, structural properties, and statistical metrics
- **⚡ Automated Workflows:** Snakemake integration for batch processing
- **🧪 Comprehensive Testing:** Full test suite with example data

## 📦 Installation

### Quick Start with uv (Recommended)
```bash
# Navigate to project directory
cd /path/to/pyNB

# For development with all features
uv pip install -e ".[dev,workflow]"
```

### Alternative Installation
```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate  
pip install -e ".[dev]"            # Core functionality + testing
pip install -e ".[workflow]"       # + Snakemake & SCENIC+ integration
```

### iPython Setup
```python
# Add this to start of iPython sessions for direct imports
import sys
sys.path.insert(0, '/path/to/pyNB/src')

# Now all imports work directly:
from analyze.Data import Data
from datastruct.Network import Network
from methods.lasso import Lasso
from bootstrap.nb_fdr import NetworkBootstrap
from analyze.CompareModels import CompareModels
```

## 🎯 Complete Usage Examples

### 1. 🌐 Network Data Import and Analysis

```python
import sys
sys.path.insert(0, 'src')  # For direct imports

from analyze.Data import Data
from datastruct.Network import Network
from methods.lasso import Lasso
from analyze.CompareModels import CompareModels
import numpy as np

# Download real dataset from public repository
print("📥 Loading dataset from JSON URL...")
dataset = Data.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json'
)

print(f"✅ Dataset loaded: {dataset.dataset}")
print(f"   📊 Expression matrix shape: {dataset.data.Y.shape}")
print(f"   🧬 Number of genes: {dataset.data.N}")
print(f"   🔬 Number of samples: {dataset.data.M}")

# Download reference (true) network
print("\n📥 Loading reference network...")
true_net = Network.from_json_url(
    'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json'
)

print(f"✅ Reference network loaded: {true_net.network}")
print(f"   🕸️ Network size: {true_net.A.shape}")
print(f"   🔗 Total edges: {np.sum(true_net.A != 0)}")
print(f"   📈 Network density: {np.sum(true_net.A != 0) / (true_net.A.shape[0] * true_net.A.shape[1]):.3f}")
```

### 2. 🔬 Network Inference with Multiple Methods

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

# Infer network from custom data
custom_lasso, custom_alpha = Lasso(custom_dataset)
custom_network = Network(custom_lasso)

print(f"\n🎯 Custom Network Inference:")
print(f"   α parameter: {custom_alpha:.6f}")
print(f"   🕸️ Edges: {np.sum(custom_network.A != 0)}")
print(f"   📈 Density: {np.sum(custom_network.A != 0) / custom_network.A.size:.3f}")

# Network density over different thresholds
thresholds = [0.0, 0.1, 0.2, 0.5, 1.0]
print(f"\n📊 Network Density vs Threshold:")
for thresh in thresholds:
    edges = np.sum(np.abs(custom_network.A) > thresh)
    density = edges / custom_network.A.size
    print(f"   Threshold {thresh:.1f}: {edges:3d} edges ({density:.3f} density)")
```

## 🏗️ Package Architecture

```
pyNB/
├── 📦 Core Components
│   ├── src/bootstrap/nb_fdr.py      # 🧮 Main NB-FDR implementation
│   ├── src/bootstrap/utils.py       # 🔧 Network analysis utilities  
│   ├── src/analyze/Data.py          # 📊 Data loading and analysis
│   ├── src/analyze/CompareModels.py # 📈 Network comparison metrics
│   └── src/datastruct/Network.py    # 🕸️ Network data structures
├── 🔬 Inference Methods  
│   ├── src/methods/lasso.py         # 🎯 LASSO regression
│   └── src/methods/lsco.py          # 📐 Least squares 
├── 🧪 Testing & Examples
│   ├── tests/                       # 🧪 Comprehensive test suite
│   ├── examples/                    # 📖 Usage examples
│   └── output/                      # 📁 Generated results
└── 📋 Configuration
    ├── pyproject.toml               # 📦 Package configuration
    └── requirements.txt             # 📋 Dependencies
```

## 🧪 Testing Your Installation

```python
# Quick verification script
import sys
sys.path.insert(0, 'src')

try:
    from analyze.Data import Data
    from datastruct.Network import Network  
    from methods.lasso import Lasso
    from bootstrap.nb_fdr import NetworkBootstrap
    from analyze.CompareModels import CompareModels
    print("✅ All imports successful!")
    
    # Test basic functionality
    nb = NetworkBootstrap()
    print("✅ NetworkBootstrap initialized")
    
    print("🎉 Installation verified - ready to use!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to run: sys.path.insert(0, 'src')")
```

## 🚀 Quick Start Script

Save this as `quickstart.py`:

```python
#!/usr/bin/env python3
"""PyNB Quick Start - Complete workflow demonstration"""

import sys
sys.path.insert(0, 'src')

from analyze.Data import Data
from datastruct.Network import Network
from methods.lasso import Lasso
from bootstrap.nb_fdr import NetworkBootstrap
from analyze.CompareModels import CompareModels
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    print("🚀 PyNB Complete Workflow Demo")
    print("=" * 35)
    
    # 1. Load real data
    print("📥 Loading example dataset...")
    dataset = Data.from_json_url(
        'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json'
    )
    
    # 2. Load reference network
    true_net = Network.from_json_url(
        'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json'
    )
    
    # 3. Infer network
    print("🧪 Running network inference...")
    lasso_net, alpha = Lasso(dataset.data)
    lasso_net = Network(lasso_net)
    
    # 4. Compare networks
    print("📊 Evaluating performance...")
    comp = CompareModels(true_net, lasso_net)
    
    # 5. Results
    print("\n🎯 Results:")
    print(f"   F1 Score: {comp.F1[0]:.4f}")
    print(f"   MCC: {comp.MCC[0]:.4f}")
    print(f"   True edges: {np.sum(true_net.A != 0)}")
    print(f"   Inferred edges: {np.sum(lasso_net.A != 0)}")
    
    print("\n✅ Demo complete! Check the full README for advanced usage.")

if __name__ == "__main__":
    main()
```

Run with: `python quickstart.py`

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
