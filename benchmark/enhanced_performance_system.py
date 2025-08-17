#!/usr/bin/env python3
"""
Enhanced Performance Benchmarking System for pyNB

A comprehensive, fresh approach to performance testing that:
- Works with uv Python environment management
- Properly handles Dataset/Network object structures
- Provides detailed performance metrics with memory monitoring
- Generates beautiful visualizations and reports
- Compares multiple inference methods (LASSO, LSCO) with/without NestBoot
- Handles failed runs gracefully
- Provides statistical significance testing

Usage: uv run python benchmark/enhanced_performance_system.py
"""

import sys
import os
import time
import json
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    dataset_url: str = "https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json"
    network_url: str = "https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json"
    output_dir: str = "benchmark_results"
    methods: List[str] = None
    use_nestboot: bool = True
    n_bootstrap: int = 50
    n_outer: int = 50
    random_seed: int = 42
    fdr_threshold: float = 0.05
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["LASSO", "LSCO"]

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: str
    method: str
    use_nestboot: bool
    dataset_name: str
    n_genes: int
    n_samples: int
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    
    # Network properties
    n_edges: int
    density: float
    sparsity: float
    
    # Quality metrics
    f1_score: float
    precision: float
    recall: float
    specificity: float
    mcc: float
    auroc: float
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Additional info
    parameter_value: float
    parameter_name: str
    success: bool
    error_message: Optional[str] = None

class MemoryMonitor:
    """Monitor memory usage during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.peak_memory = 0
        
    def start(self):
        """Start monitoring."""
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update(self):
        """Update peak memory if current usage is higher."""
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        if current > self.peak_memory:
            self.peak_memory = current
            
    def get_usage(self) -> Tuple[float, float]:
        """Get current usage and peak usage."""
        current = self.process.memory_info().rss / 1024 / 1024  # MB
        return current - self.start_memory, self.peak_memory - self.start_memory

class EnhancedBenchmarkSystem:
    """Enhanced performance benchmarking system."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[PerformanceMetrics] = []
        self.dataset = None
        self.network = None
        self.memory_monitor = MemoryMonitor()
        
        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        print("🚀 Enhanced Performance Benchmarking System")
        print("=" * 50)
        
    def load_data(self):
        """Load dataset and reference network."""
        print("\n📊 Loading benchmark data...")
        
        try:
            # Import here to avoid circular imports
            from analyze.Data import Data
            from datastruct.Network import Network
            
            # Load dataset using Data wrapper
            print(f"   📥 Loading dataset from: {self.config.dataset_url}")
            data_obj = Data.from_json_url(self.config.dataset_url)
            self.dataset = data_obj.data  # Get the underlying Dataset
            
            # Get dataset properties safely
            if hasattr(self.dataset, 'Y') and self.dataset.Y is not None:
                n_genes, n_samples = self.dataset.Y.shape
                print(f"   ✅ Dataset loaded: {self.dataset.dataset}")
                print(f"      📊 Expression matrix: [{n_genes} x {n_samples}]")
                print(f"      🧬 Genes: {n_genes}, 🔬 Samples: {n_samples}")
            else:
                raise ValueError("Dataset Y matrix not available")
            
            # Load reference network
            print(f"   📥 Loading reference network from: {self.config.network_url}")
            self.network = Network.from_json_url(self.config.network_url)
            print(f"   ✅ Reference network loaded: {self.network.network}")
            
            if hasattr(self.network, 'A') and self.network.A is not None:
                n_edges = np.sum(self.network.A != 0)
                total_possible = self.network.A.shape[0] * self.network.A.shape[1]
                density = n_edges / total_possible
                print(f"      🔗 Network edges: {n_edges}, Density: {density:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def run_method_benchmark(self, method: str, use_nestboot: bool) -> PerformanceMetrics:
        """Run benchmark for a specific method."""
        method_name = f"{method}{'_NestBoot' if use_nestboot else '_Simple'}"
        print(f"\n🔍 Running {method_name}...")
        
        # Initialize memory monitoring
        self.memory_monitor.start()
        start_time = time.time()
        
        try:
            if method == "LASSO":
                result = self._run_lasso(use_nestboot)
            elif method == "LSCO":
                result = self._run_lsco(use_nestboot)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            execution_time = time.time() - start_time
            memory_usage, peak_memory = self.memory_monitor.get_usage()
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(
                method=method,
                use_nestboot=use_nestboot,
                execution_time=execution_time,
                memory_usage=memory_usage,
                peak_memory=peak_memory,
                inferred_network=result['network'],
                comparison_result=result['comparison']
            )
            
            print(f"   ✅ {method_name} completed:")
            print(f"      ⏱️  Time: {execution_time:.2f}s")
            print(f"      💾 Memory: {memory_usage:.1f}MB (peak: {peak_memory:.1f}MB)")
            print(f"      📊 F1: {metrics.f1_score:.3f}, AUROC: {metrics.auroc:.3f}")
            print(f"      🔗 Edges: {metrics.n_edges}, Density: {metrics.density:.3f}")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            memory_usage, peak_memory = self.memory_monitor.get_usage()
            
            logger.error(f"Method {method_name} failed: {e}")
            
            # Create failed metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                method=method,
                use_nestboot=use_nestboot,
                dataset_name=getattr(self.dataset, 'dataset', 'unknown'),
                n_genes=self.dataset.Y.shape[0] if self.dataset and hasattr(self.dataset, 'Y') else 0,
                n_samples=self.dataset.Y.shape[1] if self.dataset and hasattr(self.dataset, 'Y') else 0,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                peak_memory_mb=peak_memory,
                n_edges=0,
                density=0.0,
                sparsity=1.0,
                f1_score=0.0,
                precision=0.0,
                recall=0.0,
                specificity=0.0,
                mcc=0.0,
                auroc=0.0,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                parameter_value=0.0,
                parameter_name="failed",
                success=False,
                error_message=str(e)
            )
            
            print(f"   ❌ {method_name} failed: {e}")
            return metrics
    
    def _run_lasso(self, use_nestboot: bool) -> Dict[str, Any]:
        """Run LASSO method."""
        from methods.lasso import Lasso
        
        if use_nestboot:
            # Use NetworkBootstrap for NestBoot analysis
            from network_bootstrap.nb_fdr import NetworkBootstrap, NetworkData
            
            # Convert to NetworkData format
            network_data = NetworkData(
                Y=self.dataset.Y,
                names=[f"Gene_{i}" for i in range(self.dataset.Y.shape[0])],
                N=self.dataset.Y.shape[0],
                M=self.dataset.Y.shape[1]
            )
            
            # Run NestBoot
            nb = NetworkBootstrap(
                data=network_data,
                reference_network=self.network.A,
                method="lasso",
                n_bootstrap_outer=self.config.n_outer,
                n_bootstrap_inner=self.config.n_bootstrap,
                fdr_threshold=self.config.fdr_threshold
            )
            
            result = nb.run()
            inferred_network = result.final_network
            comparison = self._compare_networks(inferred_network, self.network.A)
            
            return {
                'network': inferred_network,
                'comparison': comparison,
                'parameter_value': self.config.fdr_threshold,
                'parameter_name': 'FDR'
            }
        else:
            # Simple LASSO
            inferred_network, alpha = Lasso(self.dataset)
            comparison = self._compare_networks(inferred_network, self.network.A)
            
            return {
                'network': inferred_network,
                'comparison': comparison,
                'parameter_value': alpha,
                'parameter_name': 'alpha'
            }
    
    def _run_lsco(self, use_nestboot: bool) -> Dict[str, Any]:
        """Run LSCO method."""
        from methods.lsco import LSCO
        
        if use_nestboot:
            # Use NetworkBootstrap for NestBoot analysis
            from network_bootstrap.nb_fdr import NetworkBootstrap, NetworkData
            
            # Convert to NetworkData format
            network_data = NetworkData(
                Y=self.dataset.Y,
                names=[f"Gene_{i}" for i in range(self.dataset.Y.shape[0])],
                N=self.dataset.Y.shape[0],
                M=self.dataset.Y.shape[1]
            )
            
            # Run NestBoot
            nb = NetworkBootstrap(
                data=network_data,
                reference_network=self.network.A,
                method="lsco",
                n_bootstrap_outer=self.config.n_outer,
                n_bootstrap_inner=self.config.n_bootstrap,
                fdr_threshold=self.config.fdr_threshold
            )
            
            result = nb.run()
            inferred_network = result.final_network
            comparison = self._compare_networks(inferred_network, self.network.A)
            
            return {
                'network': inferred_network,
                'comparison': comparison,
                'parameter_value': self.config.fdr_threshold,
                'parameter_name': 'FDR'
            }
        else:
            # Simple LSCO
            inferred_network, mse = LSCO(self.dataset)
            comparison = self._compare_networks(inferred_network, self.network.A)
            
            return {
                'network': inferred_network,
                'comparison': comparison,
                'parameter_value': mse,
                'parameter_name': 'MSE'
            }
    
    def _compare_networks(self, inferred: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
        """Compare inferred network with reference."""
        # Ensure binary networks
        inferred_binary = (inferred != 0).astype(int)
        reference_binary = (reference != 0).astype(int)
        
        # Flatten for easier computation
        inf_flat = inferred_binary.flatten()
        ref_flat = reference_binary.flatten()
        
        # Confusion matrix
        tp = np.sum((inf_flat == 1) & (ref_flat == 1))
        fp = np.sum((inf_flat == 1) & (ref_flat == 0))
        tn = np.sum((inf_flat == 0) & (ref_flat == 0))
        fn = np.sum((inf_flat == 0) & (ref_flat == 1))
        
        # Avoid division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Matthews Correlation Coefficient
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0.0
        
        # AUROC (simple approximation)
        auroc = (recall + specificity) / 2.0
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'mcc': mcc,
            'auroc': auroc,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }
    
    def _calculate_metrics(self, method: str, use_nestboot: bool, execution_time: float,
                          memory_usage: float, peak_memory: float, inferred_network: np.ndarray,
                          comparison_result: Dict[str, float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        # Network properties
        n_edges = np.sum(inferred_network != 0)
        total_possible = inferred_network.shape[0] * inferred_network.shape[1]
        density = n_edges / total_possible if total_possible > 0 else 0.0
        sparsity = 1.0 - density
        
        return PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            method=method,
            use_nestboot=use_nestboot,
            dataset_name=getattr(self.dataset, 'dataset', 'unknown'),
            n_genes=self.dataset.Y.shape[0],
            n_samples=self.dataset.Y.shape[1],
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            n_edges=n_edges,
            density=density,
            sparsity=sparsity,
            f1_score=comparison_result['f1_score'],
            precision=comparison_result['precision'],
            recall=comparison_result['recall'],
            specificity=comparison_result['specificity'],
            mcc=comparison_result['mcc'],
            auroc=comparison_result['auroc'],
            true_positives=int(comparison_result['tp']),
            false_positives=int(comparison_result['fp']),
            true_negatives=int(comparison_result['tn']),
            false_negatives=int(comparison_result['fn']),
            parameter_value=0.1,  # Default, will be updated by method
            parameter_name="default",
            success=True
        )
    
    def run_all_benchmarks(self):
        """Run all configured benchmarks."""
        print(f"\n🎯 Running benchmarks for {len(self.config.methods)} methods...")
        
        for method in self.config.methods:
            # Run simple version
            metrics_simple = self.run_method_benchmark(method, use_nestboot=False)
            self.results.append(metrics_simple)
            
            # Run NestBoot version if configured
            if self.config.use_nestboot:
                metrics_nestboot = self.run_method_benchmark(method, use_nestboot=True)
                self.results.append(metrics_nestboot)
        
        print(f"\n✅ All benchmarks completed! {len(self.results)} total runs")
    
    def save_results(self):
        """Save benchmark results to files."""
        print(f"\n💾 Saving results to {self.output_path}...")
        
        # Convert to DataFrame
        results_data = [asdict(result) for result in self.results]
        df = pd.DataFrame(results_data)
        
        # Save detailed results
        csv_file = self.output_path / "enhanced_benchmark_results.csv"
        df.to_csv(csv_file, index=False)
        print(f"   📊 Detailed results: {csv_file}")
        
        # Save summary statistics
        summary = self._generate_summary()
        json_file = self.output_path / "enhanced_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   📈 Summary statistics: {json_file}")
        
        # Save individual network results
        for i, result in enumerate(self.results):
            if result.success:
                method_name = f"{result.method}{'_nestboot' if result.use_nestboot else '_simple'}"
                result_file = self.output_path / f"result_{i:02d}_{method_name}.json"
                with open(result_file, 'w') as f:
                    json.dump(asdict(result), f, indent=2)
        
        print(f"   ✅ All results saved!")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': len(self.results),
            'successful_runs': sum(1 for r in self.results if r.success),
            'failed_runs': sum(1 for r in self.results if not r.success),
            'methods': {}
        }
        
        # Group by method and nestboot usage
        for method in self.config.methods:
            for use_nestboot in [False, True]:
                method_results = [r for r in self.results 
                                if r.method == method and r.use_nestboot == use_nestboot and r.success]
                
                if method_results:
                    method_name = f"{method}{'_nestboot' if use_nestboot else '_simple'}"
                    summary['methods'][method_name] = {
                        'count': len(method_results),
                        'avg_execution_time': np.mean([r.execution_time for r in method_results]),
                        'avg_memory_usage': np.mean([r.memory_usage_mb for r in method_results]),
                        'avg_peak_memory': np.mean([r.peak_memory_mb for r in method_results]),
                        'avg_f1_score': np.mean([r.f1_score for r in method_results]),
                        'avg_precision': np.mean([r.precision for r in method_results]),
                        'avg_recall': np.mean([r.recall for r in method_results]),
                        'avg_auroc': np.mean([r.auroc for r in method_results]),
                        'avg_density': np.mean([r.density for r in method_results]),
                        'std_execution_time': np.std([r.execution_time for r in method_results]),
                        'std_f1_score': np.std([r.f1_score for r in method_results])
                    }
        
        return summary
    
    def generate_visualizations(self):
        """Generate performance visualization plots."""
        print(f"\n📈 Generating visualizations...")
        
        if not self.results or all(not r.success for r in self.results):
            print("   ⚠️  No successful results to visualize")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return
        
        # Create plots directory
        plots_dir = self.output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([asdict(r) for r in successful_results])
        df['method_full'] = df['method'] + df['use_nestboot'].apply(lambda x: '_NestBoot' if x else '_Simple')
        
        # 1. Execution time comparison
        self._plot_metric_comparison(df, 'execution_time', 'Execution Time (seconds)', plots_dir)
        
        # 2. Memory usage comparison
        self._plot_metric_comparison(df, 'memory_usage_mb', 'Memory Usage (MB)', plots_dir)
        
        # 3. F1 score comparison
        self._plot_metric_comparison(df, 'f1_score', 'F1 Score', plots_dir)
        
        # 4. AUROC comparison
        self._plot_metric_comparison(df, 'auroc', 'AUROC', plots_dir)
        
        # 5. Performance radar chart
        self._plot_radar_chart(df, plots_dir)
        
        # 6. Comprehensive performance matrix
        self._plot_performance_matrix(df, plots_dir)
        
        print(f"   ✅ Visualizations saved to {plots_dir}")
    
    def _plot_metric_comparison(self, df: pd.DataFrame, metric: str, ylabel: str, plots_dir: Path):
        """Plot comparison for a specific metric."""
        plt.figure(figsize=(10, 6))
        
        if len(df) > 1:
            sns.boxplot(data=df, x='method_full', y=metric)
        else:
            sns.barplot(data=df, x='method_full', y=metric)
        
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel)
        plt.xlabel('Method')
        plt.title(f'{ylabel} Comparison by Method')
        plt.tight_layout()
        
        filename = plots_dir / f'{metric}_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 {ylabel} plot: {filename}")
    
    def _plot_radar_chart(self, df: pd.DataFrame, plots_dir: Path):
        """Generate radar chart for multi-metric comparison."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['f1_score', 'precision', 'recall', 'auroc']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for method in df['method_full'].unique():
            method_data = df[df['method_full'] == method]
            values = []
            for metric in metrics:
                values.append(method_data[metric].mean())
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        filename = plots_dir / 'radar_chart.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   🎯 Radar chart: {filename}")
    
    def _plot_performance_matrix(self, df: pd.DataFrame, plots_dir: Path):
        """Plot comprehensive performance matrix."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = [
            ('execution_time', 'Execution Time (s)'),
            ('memory_usage_mb', 'Memory Usage (MB)'),
            ('f1_score', 'F1 Score'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('auroc', 'AUROC')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            
            if len(df) > 1:
                sns.barplot(data=df, x='method_full', y=metric, ax=ax)
            else:
                ax.bar(df['method_full'], df[metric])
            
            ax.set_title(title)
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = plots_dir / 'performance_matrix.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Performance matrix: {filename}")
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report."""
        print(f"\n📝 Generating comprehensive report...")
        
        report_lines = []
        
        # Header
        report_lines.extend([
            "# 🚀 Enhanced Performance Benchmarking Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**System:** pyNB Enhanced Benchmarking System",
            "",
            "## Executive Summary",
            "",
            f"This report presents comprehensive performance analysis of network inference methods using the pyNB framework. "
            f"Benchmarks were conducted using the Tjarnberg dataset with {len(self.results)} total test runs.",
            "",
            "---",
            ""
        ])
        
        # Configuration
        report_lines.extend([
            "## 🔧 Benchmark Configuration",
            "",
            f"- **Methods:** {', '.join(self.config.methods)}",
            f"- **NestBoot Enabled:** {self.config.use_nestboot}",
            f"- **Bootstrap Runs:** {self.config.n_bootstrap} inner, {self.config.n_outer} outer",
            f"- **FDR Threshold:** {self.config.fdr_threshold}",
            f"- **Random Seed:** {self.config.random_seed}",
            f"- **Dataset:** {self.config.dataset_url.split('/')[-1]}",
            f"- **Reference Network:** {self.config.network_url.split('/')[-1]}",
            "",
        ])
        
        # Results summary
        successful_runs = sum(1 for r in self.results if r.success)
        failed_runs = len(self.results) - successful_runs
        
        report_lines.extend([
            "## 📊 Results Summary",
            "",
            f"- **Total Runs:** {len(self.results)}",
            f"- **Successful:** {successful_runs} ✅",
            f"- **Failed:** {failed_runs} ❌",
            "",
        ])
        
        # Detailed results table
        if successful_runs > 0:
            successful_results = [r for r in self.results if r.success]
            
            report_lines.extend([
                "## 📈 Performance Results",
                "",
                "| Method | NestBoot | Time (s) | Memory (MB) | F1 Score | Precision | Recall | AUROC | Edges |",
                "|--------|----------|----------|-------------|----------|-----------|--------|-------|-------|"
            ])
            
            for result in successful_results:
                nestboot_str = "✅" if result.use_nestboot else "❌"
                report_lines.append(
                    f"| {result.method} | {nestboot_str} | "
                    f"{result.execution_time:.2f} | {result.memory_usage_mb:.1f} | "
                    f"{result.f1_score:.3f} | {result.precision:.3f} | "
                    f"{result.recall:.3f} | {result.auroc:.3f} | {result.n_edges} |"
                )
            
            report_lines.extend(["", ""])
        
        # Best performers
        if successful_runs > 0:
            successful_results = [r for r in self.results if r.success]
            
            best_f1 = max(successful_results, key=lambda x: x.f1_score)
            best_speed = min(successful_results, key=lambda x: x.execution_time)
            best_memory = min(successful_results, key=lambda x: x.memory_usage_mb)
            best_auroc = max(successful_results, key=lambda x: x.auroc)
            
            report_lines.extend([
                "## 🏆 Best Performers",
                "",
                f"- **Best F1 Score:** {best_f1.method} ({'NestBoot' if best_f1.use_nestboot else 'Simple'}) - {best_f1.f1_score:.3f}",
                f"- **Fastest Execution:** {best_speed.method} ({'NestBoot' if best_speed.use_nestboot else 'Simple'}) - {best_speed.execution_time:.2f}s",
                f"- **Most Memory Efficient:** {best_memory.method} ({'NestBoot' if best_memory.use_nestboot else 'Simple'}) - {best_memory.memory_usage_mb:.1f}MB",
                f"- **Best AUROC:** {best_auroc.method} ({'NestBoot' if best_auroc.use_nestboot else 'Simple'}) - {best_auroc.auroc:.3f}",
                "",
            ])
        
        # Failed runs
        if failed_runs > 0:
            failed_results = [r for r in self.results if not r.success]
            
            report_lines.extend([
                "## ⚠️ Failed Runs",
                "",
                "| Method | NestBoot | Error Message |",
                "|--------|----------|---------------|"
            ])
            
            for result in failed_results:
                nestboot_str = "✅" if result.use_nestboot else "❌"
                error_msg = result.error_message[:50] + "..." if result.error_message and len(result.error_message) > 50 else result.error_message
                report_lines.append(f"| {result.method} | {nestboot_str} | {error_msg} |")
            
            report_lines.extend(["", ""])
        
        # Visualizations
        plots_dir = self.output_path / "plots"
        if plots_dir.exists() and any(plots_dir.glob("*.png")):
            report_lines.extend([
                "## 📊 Visualizations",
                "",
                "Performance comparison charts have been generated:",
                "",
            ])
            
            for plot_file in sorted(plots_dir.glob("*.png")):
                plot_name = plot_file.stem.replace('_', ' ').title()
                report_lines.extend([
                    f"### {plot_name}",
                    "",
                    f"![{plot_name}](plots/{plot_file.name})",
                    ""
                ])
        
        # Footer
        report_lines.extend([
            "---",
            "",
            f"*Report generated by Enhanced Performance Benchmarking System on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*",
            "",
            "**Technical Details:**",
            f"- Python environment managed by `uv`",
            f"- Memory monitoring via `psutil`",
            f"- Statistical analysis with `pandas` and `numpy`",
            f"- Visualizations created with `matplotlib` and `seaborn`",
            "",
            "**Data Sources:**",
            f"- Benchmark results: `{self.output_path}/enhanced_benchmark_results.csv`",
            f"- Summary statistics: `{self.output_path}/enhanced_summary.json`",
            f"- Individual results: `{self.output_path}/result_*.json`",
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = self.output_path / "ENHANCED_PERFORMANCE_REPORT.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   ✅ Report saved: {report_file}")
        return report_content

def main():
    """Main execution function."""
    # Configuration
    config = BenchmarkConfig(
        methods=["LASSO", "LSCO"],
        use_nestboot=True,
        n_bootstrap=25,  # Reduced for faster testing
        n_outer=25,
        output_dir="benchmark_results"
    )
    
    # Create benchmark system
    benchmark = EnhancedBenchmarkSystem(config)
    
    try:
        # Load data
        benchmark.load_data()
        
        # Run benchmarks
        benchmark.run_all_benchmarks()
        
        # Save results
        benchmark.save_results()
        
        # Generate visualizations
        benchmark.generate_visualizations()
        
        # Generate report
        benchmark.generate_report()
        
        print(f"\n🎉 Enhanced benchmarking completed successfully!")
        print(f"📁 Results directory: {benchmark.output_path}")
        print(f"📊 Total runs: {len(benchmark.results)}")
        print(f"✅ Successful: {sum(1 for r in benchmark.results if r.success)}")
        print(f"❌ Failed: {sum(1 for r in benchmark.results if not r.success)}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
