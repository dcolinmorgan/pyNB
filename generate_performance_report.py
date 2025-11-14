#!/usr/bin/env python3
"""
Generate Performance Comparison Report from Benchmark Results.
Reads CSV/JSON output files, excludes failed runs (F1/MCC/AUROC = 0 or 1),
and creates a markdown report with mean/median comparisons.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

class PerformanceReportGenerator:
    """Generate performance comparison reports, excluding failed runs."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.python_results = {}
        self.matlab_results = {}
        self.summary_stats = {}
        self.failed_runs = {}
        
    def load_results(self):
        """Load benchmark results from files."""
        print("📊 Loading benchmark results...")
        
        if (self.results_dir / "python_benchmark_results.csv").exists():
            self.python_results['detailed'] = pd.read_csv(self.results_dir / "python_benchmark_results.csv")
            print(f"   ✅ Python detailed results: {len(self.python_results['detailed'])} entries")
        
        if (self.results_dir / "python_summary.json").exists():
            with open(self.results_dir / "python_summary.json", 'r') as f:
                self.python_results['summary'] = json.load(f)
            print(f"   ✅ Python summary: {len(self.python_results['summary'])} methods")
        
        if (self.results_dir / "matlab_benchmark_results.csv").exists():
            self.matlab_results['detailed'] = pd.read_csv(self.results_dir / "matlab_benchmark_results.csv")
            print(f"   ✅ MATLAB detailed results: {len(self.matlab_results['detailed'])} entries")
        
        if (self.results_dir / "matlab_summary.json").exists():
            with open(self.results_dir / "matlab_summary.json", 'r') as f:
                self.matlab_results['summary'] = json.load(f)
            print(f"   ✅ MATLAB summary: {len(self.matlab_results['summary'])} methods")
    
    def analyze_performance(self):
        """Analyze performance metrics, excluding failed runs."""
        print("🔍 Analyzing performance metrics...")
        
        all_data = []
        
        if 'detailed' in self.python_results:
            python_df = self.python_results['detailed'].copy()
            python_df['platform'] = 'Python'
            python_df['platform_method'] = python_df['platform'] + '_' + python_df['method']
            all_data.append(python_df)
        
        if 'detailed' in self.matlab_results:
            matlab_df = self.matlab_results['detailed'].copy()
            matlab_df['platform'] = 'MATLAB'
            matlab_df['platform_method'] = matlab_df['platform'] + '_' + matlab_df['method_name']
            all_data.append(matlab_df)
        
        if all_data:
            self.combined_data = pd.concat(all_data, ignore_index=True)
            
            # Spot check for failed runs (F1, MCC, AUROC = 0 or 1)
            self.failed_runs = self._identify_failed_runs()
            print(f"   ⚠️ Detected {len(self.failed_runs['indices'])} failed runs")
            
            # Filter out failed runs
            self.combined_data = self.combined_data[~self.combined_data.index.isin(self.failed_runs['indices'])]
            
            # Calculate summary statistics
            self.summary_stats = self._calculate_summary_stats()
            print(f"   ✅ Combined analysis: {len(self.combined_data)} valid entries")
        else:
            print("   ⚠️ No detailed data available for analysis")
    
    def _identify_failed_runs(self) -> Dict[str, Any]:
        """Identify runs where F1, MCC, or AUROC are exactly 0 or 1."""
        failed = {'indices': [], 'details': []}
        metrics = ['f1_score', 'mcc', 'auroc']
        
        for idx, row in self.combined_data.iterrows():
            for metric in metrics:
                if metric in row and (row[metric] == 0.0 or row[metric] == 1.0):
                    failed['indices'].append(idx)
                    failed['details'].append({
                        'platform': row['platform'],
                        'method': row['platform_method'],
                        'metric': metric,
                        'value': row[metric],
                        'timestamp': row['timestamp']
                    })
                    break  # One failure per run is enough
        
        return failed
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics (mean and median) for valid runs."""
        stats = {}
        
        # Group by platform and method
        grouped = self.combined_data.groupby(['platform', 'method']).agg({
            'execution_time': ['mean', 'median', 'std', 'min', 'max'],
            'memory_usage': ['mean', 'median', 'std', 'min', 'max'],
            'f1_score': ['mean', 'median', 'std', 'min', 'max'],
            'precision': ['mean', 'median', 'std', 'min', 'max'],
            'sensitivity': ['mean', 'median', 'std', 'min', 'max'],
            'sparsity': ['mean', 'median', 'std', 'min', 'max'],
            'density': ['mean', 'median', 'std', 'min', 'max'],
            'auroc': ['mean', 'median', 'std', 'min', 'max']
        }).round(4)
        
        stats['grouped_metrics'] = grouped
        
        # Find best performers (excluding failed runs)
        best_performers = {}
        for metric in ['execution_time', 'memory_usage', 'f1_score', 'precision', 'sensitivity', 'auroc']:
            if metric in ['execution_time', 'memory_usage']:
                best_idx = self.combined_data[metric].idxmin()
            else:
                best_idx = self.combined_data[metric].idxmax()
            
            if pd.notna(best_idx):
                best_row = self.combined_data.loc[best_idx]
                best_performers[metric] = {
                    'platform': best_row['platform'],
                    'method': best_row['method'],
                    'value': best_row[metric],
                    'platform_method': best_row['platform_method']
                }
        
        stats['best_performers'] = best_performers
        
        # Calculate scalability metrics
        if 'dataset_size' in self.combined_data.columns:
            scalability = self.combined_data.groupby(['platform_method', 'dataset_size']).agg({
                'execution_time': ['mean', 'median'],
                'memory_usage': ['mean', 'median']
            }).round(2)
            stats['scalability'] = scalability
        
        return stats
    
    def generate_visualizations(self):
        """Generate performance comparison visualizations for valid runs."""
        print("📈 Generating visualizations...")
        
        if not hasattr(self, 'combined_data') or self.combined_data.empty:
            print("   ⚠️ No valid data available for visualization")
            return
        
        plt.style.use('default')
        sns.set_palette("husl")
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        self._plot_metric_comparison('execution_time', 'Execution Time (seconds)', plots_dir)
        self._plot_metric_comparison('memory_usage', 'Memory Usage (MB)', plots_dir)
        self._plot_metric_comparison('f1_score', 'F1 Score', plots_dir)
        self._plot_metric_comparison('auroc', 'AUROC', plots_dir)
        self._plot_radar_chart(plots_dir)
        
        print(f"   ✅ Visualizations saved to {plots_dir}")
    
    def _plot_metric_comparison(self, metric: str, ylabel: str, plots_dir: Path):
        """Plot comparison for a specific metric."""
        plt.figure(figsize=(10, 6))
        data_to_plot = self.combined_data.dropna(subset=[metric])
        if not data_to_plot.empty:
            sns.boxplot(data=data_to_plot, x='platform_method', y=metric)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(ylabel)
            plt.xlabel('Platform and Method')
            plt.title(f'{ylabel} Comparison by Platform and Method')
            plt.tight_layout()
            plt.savefig(plots_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_radar_chart(self, plots_dir: Path):
        """Generate radar chart for multi-metric comparison."""
        if not hasattr(self, 'summary_stats') or 'best_performers' not in self.summary_stats:
            return
        
        methods = self.combined_data['platform_method'].unique()
        metrics = ['f1_score', 'precision', 'sensitivity', 'auroc']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for method in methods:
            method_data = self.combined_data[self.combined_data['platform_method'] == method]
            if not method_data.empty:
                values = []
                for metric in metrics:
                    values.append(method_data[metric].mean())
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=2, label=method)
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.savefig(plots_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_markdown_report(self) -> str:
        """Generate markdown performance report, including failed runs."""
        print("📝 Generating markdown report...")
        
        report = []
        report.extend(self._generate_header())
        report.extend(self._generate_executive_summary())
        report.extend(self._generate_failed_runs_section())
        report.extend(self._generate_detailed_analysis())
        report.extend(self._generate_visualizations_section())
        report.extend(self._generate_best_performers_section())
        report.extend(self._generate_recommendations())
        report.extend(self._generate_footer())
        
        return '\n'.join(report)
    
    def _generate_header(self) -> List[str]:
        """Generate report header."""
        return [
            "# 📊 Performance Comparison Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            "This report compares Python pyNB and MATLAB GeneSPIDER2 performance, excluding failed runs (F1/MCC/AUROC = 0 or 1). Results are based on benchmark runs using the Tjarnberg dataset.",
            "",
            "---",
            ""
        ]
    
    def _generate_executive_summary(self) -> List[str]:
        """Generate executive summary section."""
        summary = []
        
        if hasattr(self, 'combined_data') and not self.combined_data.empty:
            total_tests = len(self.combined_data) + len(self.failed_runs['indices'])
            valid_tests = len(self.combined_data)
            platforms = self.combined_data['platform'].unique()
            methods = self.combined_data['method'].unique()
            
            summary.extend([
                "## 🎯 Test Configuration",
                "",
                f"- **Total Tests:** {total_tests}",
                f"- **Valid Tests:** {valid_tests}",
                f"- **Failed Tests:** {len(self.failed_runs['indices'])}",
                f"- **Platforms:** {', '.join(platforms)}",
                f"- **Methods:** {', '.join(methods)}",
                f"- **Dataset:** Tjarnberg-ID252384 (N50, E150, SNR10)",
                ""
            ])
        
        return summary
    
    def _generate_failed_runs_section(self) -> List[str]:
        """Generate section for failed runs."""
        section = [
            "## ⚠️ Failed Runs",
            "",
            "Runs with F1, MCC, or AUROC exactly 0 or 1 are considered failed and excluded from averages/medians.",
            ""
        ]
        
        if self.failed_runs['details']:
            section.extend([
                "| Platform | Method | Metric | Value | Timestamp |",
                "|----------|--------|--------|-------|-----------|"
            ])
            for run in self.failed_runs['details']:
                section.append(
                    f"| {run['platform']} | {run['method']} | {run['metric']} | {run['value']} | {run['timestamp']} |"
                )
        else:
            section.append("*No failed runs detected.*")
        
        section.extend(["", ""])
        return section
    
    def _generate_detailed_analysis(self) -> List[str]:
        """Generate detailed performance analysis section."""
        analysis = [
            "## 📈 Detailed Performance Analysis",
            ""
        ]
        
        if not hasattr(self, 'summary_stats') or 'grouped_metrics' not in self.summary_stats:
            analysis.append("*No detailed performance data available.*")
            return analysis
        
        analysis.extend([
            "### Performance Metrics Summary (Mean / Median)",
            "",
            "| Platform | Method | Time (s) | Memory (MB) | F1 Score | Precision | Recall | AUROC |",
            "|----------|--------|----------|-------------|----------|-----------|--------|-------|"
        ])
        
        grouped = self.summary_stats['grouped_metrics']
        for (platform, method), metrics in grouped.iterrows():
            analysis.append(
                f"| {platform} | {method} | "
                f"{metrics[('execution_time', 'mean')]:.2f} / {metrics[('execution_time', 'median')]:.2f} | "
                f"{metrics[('memory_usage', 'mean')]:.1f} / {metrics[('memory_usage', 'median')]:.1f} | "
                f"{metrics[('f1_score', 'mean')]:.3f} / {metrics[('f1_score', 'median')]:.3f} | "
                f"{metrics[('precision', 'mean')]:.3f} / {metrics[('precision', 'median')]:.3f} | "
                f"{metrics[('sensitivity', 'mean')]:.3f} / {metrics[('sensitivity', 'median')]:.3f} | "
                f"{metrics[('auroc', 'mean')]:.3f} / {metrics[('auroc', 'median')]:.3f} |"
            )
        
        analysis.extend(["", ""])
        return analysis
    
    def _generate_visualizations_section(self) -> List[str]:
        """Generate visualizations section."""
        viz_section = [
            "## 📊 Performance Visualizations",
            "",
            "Charts compare valid runs across platforms and methods:",
            ""
        ]
        
        plots_dir = self.results_dir / "plots"
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                for plot_file in sorted(plot_files):
                    plot_name = plot_file.stem.replace('_', ' ').title()
                    viz_section.extend([
                        f"### {plot_name}",
                        "",
                        f"![{plot_name}](plots/{plot_file.name})",
                        ""
                    ])
            else:
                viz_section.append("*No visualization files found.*")
        else:
            viz_section.append("*Plots directory not found.*")
        
        return viz_section
    
    def _generate_best_performers_section(self) -> List[str]:
        """Generate best performers section."""
        best_section = [
            "## 🏆 Best Performers",
            ""
        ]
        
        if 'best_performers' in self.summary_stats:
            best_performers = self.summary_stats['best_performers']
            
            best_section.extend([
                "| Metric | Winner | Platform | Method | Value |",
                "|--------|--------|----------|--------|-------|"
            ])
            
            for metric, info in best_performers.items():
                metric_name = metric.replace('_', ' ').title()
                best_section.append(
                    f"| {metric_name} | 🥇 | {info['platform']} | {info['method']} | {info['value']:.3f} |"
                )
            
            best_section.extend(["", ""])
        else:
            best_section.append("*No performance comparison data available.*")
        
        return best_section
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations section."""
        recommendations = [
            "## 🎯 Recommendations",
            "",
            "Based on valid benchmark results:",
            ""
        ]
        
        if 'best_performers' in self.summary_stats:
            best = self.summary_stats['best_performers']
            
            if 'execution_time' in best:
                speed_winner = best['execution_time']
                recommendations.append(
                    f"- **Fastest Execution:** {speed_winner['platform']} {speed_winner['method']} "
                    f"({speed_winner['value']:.2f} seconds)"
                )
            
            if 'memory_usage' in best:
                memory_winner = best['memory_usage']
                recommendations.append(
                    f"- **Most Memory Efficient:** {memory_winner['platform']} {memory_winner['method']} "
                    f"({memory_winner['value']:.1f} MB)"
                )
            
            if 'f1_score' in best:
                accuracy_winner = best['f1_score']
                recommendations.append(
                    f"- **Best Network Quality (F1):** {accuracy_winner['platform']} {accuracy_winner['method']} "
                    f"(F1 Score: {accuracy_winner['value']:.3f})"
                )
            
            if 'auroc' in best:
                auroc_winner = best['auroc']
                recommendations.append(
                    f"- **Best Network Quality (AUROC):** {auroc_winner['platform']} {auroc_winner['method']} "
                    f"(AUROC: {auroc_winner['value']:.3f})"
                )
            
            recommendations.extend(["", ""])
        
        return recommendations
    
    def _generate_footer(self) -> List[str]:
        """Generate report footer."""
        return [
            "---",
            "",
            f"*Report generated automatically on {datetime.now().strftime('%Y-%m-%d')}*",
            "",
            "**Data Sources:**",
            f"- Python results: `{self.results_dir}/python_benchmark_results.csv`",
            f"- MATLAB results: `{self.results_dir}/matlab_benchmark_results.csv`",
            "",
            "**Methodology:**",
            "- Failed runs (F1/MCC/AUROC = 0 or 1) excluded from averages/medians",
            "- Tests run with Tjarnberg dataset (N50, E150, SNR10)",
            "- Memory usage measured using platform-specific tools",
            "- Execution time measured as wall-clock time",
            "- Network quality metrics calculated using standard confusion matrix methods"
        ]
    
    def save_report(self, filename: str = "PERFORMANCE_COMPARISON_GENERATED.md"):
        """Save the generated report to file."""
        report_content = self.generate_markdown_report()
        output_file = Path(filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"✅ Performance report saved to: {output_file}")
        return output_file

def main():
    """Main execution function."""
    print("🚀 Performance Report Generator")
    print("=" * 40)
    
    generator = PerformanceReportGenerator()
    generator.load_results()
    generator.analyze_performance()
    generator.generate_visualizations()
    output_file = generator.save_report()
    
    print(f"\n🎉 Performance comparison report generated!")
    print(f"📄 Report file: {output_file}")
    
    if hasattr(generator, 'combined_data') and not generator.combined_data.empty:
        print(f"📊 Analysis included {len(generator.combined_data)} valid benchmark results")
        print(f"   ⚠️ Excluded {len(generator.failed_runs['indices'])} failed runs")
        platforms = generator.combined_data['platform'].unique()
        methods = generator.combined_data['method'].unique()
        print(f"🔧 Platforms: {', '.join(platforms)}")
        print(f"⚙️ Methods: {', '.join(methods)}")

if __name__ == "__main__":
    main()
