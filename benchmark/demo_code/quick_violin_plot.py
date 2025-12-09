#!/usr/bin/env python3
"""
Quick Violin Plot Comparison: Python vs MATLAB Benchmark Results

A simplified version for quick analysis and plotting.
Run this after installing dependencies: pip install pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_comparison_plot():
    """Create the main violin plot comparison."""

    # Load data
    base_dir = Path(__file__).parent.parent / "nestboot_results"

    # Load Python results
    python_df = pd.read_csv(base_dir / "python_snr_all_methods_results.csv")
    python_df['language'] = 'Python'
    python_df['snr'] = python_df['dataset'].str.extract(r'SNR([0-9.]+)', expand=False)
    python_df['method'] = python_df['method'].str.upper()

    # Load MATLAB results
    matlab_df = pd.read_csv(base_dir / "matlab_snr_all_methods_results.csv")
    matlab_df['language'] = 'MATLAB'
    matlab_df['snr'] = matlab_df['dataset'].str.extract(r'SNR([0-9.]+)', expand=False)
    matlab_df['method'] = matlab_df['method'].str.upper()

    # Load nestboot results
    nestboot_df = pd.read_csv(base_dir / "matlab_nestboot_results.csv")
    nestboot_df['language'] = 'MATLAB'
    nestboot_df['snr'] = nestboot_df['dataset'].str.extract(r'SNR([0-9.]+)', expand=False)
    nestboot_df['method'] = 'NESTBOOT+' + nestboot_df['method'].str.upper()

    # Combine data
    data = pd.concat([
        python_df[['method', 'snr', 'language', 'f1_score', 'auroc', 'mcc']].rename(columns={'f1_score': 'f1'}),
        matlab_df[['method', 'snr', 'language', 'f1_score', 'auroc', 'mcc']].rename(columns={'f1_score': 'f1'}),
        nestboot_df[['method', 'snr', 'language', 'f1', 'auroc', 'mcc']]
    ], ignore_index=True)

    # Filter to desired methods
    methods = ['LASSO', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS', 'NESTBOOT+LASSO', 'NESTBOOT+LSCO']
    data = data[data['method'].isin(methods)].copy()

    # Convert SNR to string for categorical plotting
    data['snr'] = data['snr'].astype(str)

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle('Python vs MATLAB Benchmark Comparison\nViolin Plots by Method and SNR',
                 fontsize=16, fontweight='bold', y=0.98)

    metrics = ['f1', 'auroc', 'mcc']
    labels = ['F1 Score', 'AUROC', 'MCC']

    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]

        sns.violinplot(
            data=data,
            x='method',
            y=metric,
            hue='language',
            split=True,
            inner='quartile',
            palette=['#1f77b4', '#ff7f0e'],
            ax=ax,
            order=methods,
            cut=0,
            scale='width'
        )

        ax.set_title(f'{label} Distribution by Method', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.legend().set_visible(False)

    plt.tight_layout()

    # Save plot
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'python_matlab_violin_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved to: {output_dir}/python_matlab_violin_comparison.png")

    # Print summary statistics
    summary = data.groupby(['method', 'language']).agg({
        'f1': ['mean', 'std', 'count'],
        'auroc': ['mean', 'std'],
        'mcc': ['mean', 'std']
    }).round(3)

    print("\nSummary Statistics:")
    print(summary.to_string())

if __name__ == "__main__":
    create_comparison_plot()
