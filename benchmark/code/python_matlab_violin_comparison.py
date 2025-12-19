#!/usr/bin/env python3
"""
Violin Plot Comparison: Python vs MATLAB Benchmark Results

Creates violin plots comparing MCC, F1, and AUROC scores between Python and MATLAB
implementations across different methods and SNR levels.

Methods compared:
- Regular: LASSO, LSCO, CLR, GENIE3, TIGRESS
- Bootstrap-enhanced: NestBoot+LASSO, NestBoot+LSCO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_process_data():
    """Load and process benchmark data from both Python and MATLAB results."""

    # Define file paths
    base_dir = Path(__file__).parent.parent / "nestboot_results"
    python_file = base_dir / "python_snr_all_methods_results.csv"
    matlab_file = base_dir / "matlab_snr_all_methods_results.csv"
    nestboot_file = base_dir / "matlab_nestboot_results.csv"

    # Load Python results
    python_df = pd.read_csv(python_file)
    python_df['language'] = 'Python'

    # Load MATLAB regular methods results
    matlab_df = pd.read_csv(matlab_file)
    matlab_df['language'] = 'MATLAB'

    # Load MATLAB nestboot results
    nestboot_df = pd.read_csv(nestboot_file)
    nestboot_df['language'] = 'MATLAB'

    # Process Python data
    python_processed = process_python_data(python_df)

    # Process MATLAB data
    matlab_processed = process_matlab_data(matlab_df, nestboot_df)

    # Combine all data
    combined_df = pd.concat([python_processed, matlab_processed], ignore_index=True)

    return combined_df

def process_python_data(df):
    """Process Python benchmark results."""
    # Extract SNR from dataset name
    df['snr'] = df['dataset'].str.extract(r'SNR([0-9.]+)', expand=False)
    df['snr'] = pd.to_numeric(df['snr'], errors='coerce')

    # Standardize method names
    df['method'] = df['method'].str.upper()

    # Rename columns for consistency
    df = df.rename(columns={
        'f1_score': 'f1',
        'auroc': 'auroc',
        'mcc': 'mcc'
    })

    # Select relevant columns
    columns = ['method', 'snr', 'language', 'f1', 'auroc', 'mcc']
    df = df[columns].copy()

    return df

def process_matlab_data(regular_df, nestboot_df):
    """Process MATLAB benchmark results."""
    # Process regular methods
    regular_df['snr'] = regular_df['dataset'].str.extract(r'SNR([0-9.]+)', expand=False)
    regular_df['snr'] = pd.to_numeric(regular_df['snr'], errors='coerce')

    # Standardize method names
    regular_df['method'] = regular_df['method'].str.upper()

    # Rename columns for consistency
    regular_df = regular_df.rename(columns={
        'f1_score': 'f1',
        'auroc': 'auroc',
        'mcc': 'mcc'
    })

    # Process nestboot results
    nestboot_df['snr'] = nestboot_df['dataset'].str.extract(r'SNR([0-9.]+)', expand=False)
    nestboot_df['snr'] = pd.to_numeric(nestboot_df['snr'], errors='coerce')

    # Create bootstrap method names
    nestboot_df['method'] = 'NESTBOOT+' + nestboot_df['method'].str.upper()

    # Select relevant columns for regular methods
    regular_columns = ['method', 'snr', 'language', 'f1', 'auroc', 'mcc']
    regular_processed = regular_df[regular_columns].copy()

    # Select relevant columns for nestboot methods
    nestboot_columns = ['method', 'snr', 'language', 'f1', 'auroc', 'mcc']
    nestboot_processed = nestboot_df[nestboot_columns].copy()

    # Combine regular and nestboot MATLAB results
    matlab_combined = pd.concat([regular_processed, nestboot_processed], ignore_index=True)

    return matlab_combined

def create_violin_plots(data, output_dir):
    """Create violin plots comparing Python vs MATLAB across methods and SNRs."""

    # Define the methods to include (in desired order)
    methods = ['LASSO', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS', 'NESTBOOT+LASSO', 'NESTBOOT+LSCO']

    # Filter data to include only these methods
    data_filtered = data[data['method'].isin(methods)].copy()

    # Convert SNR to categorical for better plotting
    data_filtered['snr'] = data_filtered['snr'].astype(str)

    # Define metrics to plot
    metrics = ['f1', 'auroc', 'mcc']
    metric_labels = ['F1 Score', 'AUROC', 'MCC']

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle('Python vs MATLAB Benchmark Comparison\nViolin Plots by Method and SNR',
                 fontsize=16, fontweight='bold', y=0.98)

    # Create violin plots for each metric
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]

        # Create the violin plot
        sns.violinplot(
            data=data_filtered,
            x='method',
            y=metric,
            hue='language',
            split=True,
            inner='quartile',
            palette=['#1f77b4', '#ff7f0e'],  # Blue for Python, Orange for MATLAB
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

        # Move legend to top right
        if idx == 0:
            ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
        else:
            ax.legend().set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'python_matlab_violin_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Create separate plots for each SNR level
    snr_values = sorted(data_filtered['snr'].unique())

    for snr in snr_values:
        snr_data = data_filtered[data_filtered['snr'] == snr].copy()

        if len(snr_data) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Python vs MATLAB Comparison - SNR {snr}',
                     fontsize=14, fontweight='bold')

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]

            sns.violinplot(
                data=snr_data,
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

            ax.set_title(f'{label}', fontsize=12)
            ax.set_xlabel('Method', fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            if idx == 0:
                ax.legend(title='Language', bbox_to_anchor=(1.02, 1), loc='upper left')
            else:
                ax.legend().set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / f'python_matlab_violin_snr_{snr}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_statistics(data, output_dir):
    """Create summary statistics table."""

    # Group by method, language, and calculate statistics
    summary = data.groupby(['method', 'language']).agg({
        'f1': ['mean', 'std', 'count'],
        'auroc': ['mean', 'std', 'count'],
        'mcc': ['mean', 'std', 'count']
    }).round(4)

    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Save to CSV
    summary.to_csv(output_dir / 'benchmark_summary_stats.csv', index=False)

    return summary

def main():
    """Main function to run the analysis."""

    print("Loading and processing benchmark data...")

    # Load and process data
    data = load_and_process_data()

    # Create output directory
    output_dir = Path(__file__).parent.parent / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Data loaded. Shape: {data.shape}")
    print(f"Methods found: {sorted(data['method'].unique())}")
    print(f"SNRS found: {sorted(data['snr'].unique())}")

    # Create violin plots
    print("Creating violin plots...")
    create_violin_plots(data, output_dir)

    # Create summary statistics
    print("Creating summary statistics...")
    summary = create_summary_statistics(data, output_dir)

    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print(f"Summary statistics saved to: {output_dir}/benchmark_summary_stats.csv")

    # Print summary
    print("\nSummary Statistics:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
