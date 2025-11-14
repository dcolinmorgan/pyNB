#!/usr/bin/env python3
"""
Test the Performance Report Generator with sample data.
This script creates sample benchmark data and tests the report generation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def create_sample_data():
    """Create sample benchmark data for testing."""
    print("📊 Creating sample benchmark data...")
    
    # Create output directory
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Sample Python results
    python_data = [
        {
            'timestamp': '2025-07-18T10:00:00',
            'dataset_size': 'small',
            'n_genes': 10,
            'n_samples': 20,
            'method': 'LASSO',
            'execution_time': 2.5,
            'memory_usage': 85.0,
            'parameter_value': 0.123456,
            'parameter_name': 'alpha',
            'num_edges': 85,
            'density': 0.850,
            'sparsity': 0.150,
            'f1_score': 0.785,
            'mcc': 0.721,
            'sensitivity': 0.737,
            'specificity': 0.823,
            'precision': 0.840,
            'true_positives': 42,
            'false_positives': 8,
            'true_negatives': 35,
            'false_negatives': 15
        },
        {
            'timestamp': '2025-07-18T10:05:00',
            'dataset_size': 'small',
            'n_genes': 10,
            'n_samples': 20,
            'method': 'LSCO',
            'execution_time': 1.8,
            'memory_usage': 65.0,
            'parameter_value': 0.234567,
            'parameter_name': 'mse',
            'num_edges': 95,
            'density': 0.950,
            'sparsity': 0.050,
            'f1_score': 0.800,
            'mcc': 0.742,
            'sensitivity': 0.842,
            'specificity': 0.775,
            'precision': 0.762,
            'true_positives': 48,
            'false_positives': 15,
            'true_negatives': 28,
            'false_negatives': 9
        },
        {
            'timestamp': '2025-07-18T10:10:00',
            'dataset_size': 'medium',
            'n_genes': 50,
            'n_samples': 100,
            'method': 'LASSO',
            'execution_time': 15.2,
            'memory_usage': 280.0,
            'parameter_value': 0.098765,
            'parameter_name': 'alpha',
            'num_edges': 420,
            'density': 0.168,
            'sparsity': 0.832,
            'f1_score': 0.745,
            'mcc': 0.683,
            'sensitivity': 0.698,
            'specificity': 0.865,
            'precision': 0.798,
            'true_positives': 184,
            'false_positives': 46,
            'true_negatives': 2035,
            'false_negatives': 80
        }
    ]
    
    # Sample MATLAB results
    matlab_data = [
        {
            'timestamp': '2025-07-18T10:15:00',
            'dataset_size': 'medium',
            'n_genes': 50,
            'n_samples': 150,
            'method': 'lasso',
            'use_nestboot': False,
            'method_name': 'lasso_simple',
            'execution_time': 8.5,
            'memory_usage': 150.0,
            'parameter_value': 0.05,
            'parameter_name': 'threshold',
            'num_edges': 380,
            'density': 0.152,
            'sparsity': 0.848,
            'f1_score': 0.710,
            'mcc': 0.645,
            'sensitivity': 0.667,
            'specificity': 0.875,
            'precision': 0.760,
            'true_positives': 38,
            'false_positives': 12,
            'true_negatives': 31,
            'false_negatives': 19
        },
        {
            'timestamp': '2025-07-18T10:20:00',
            'dataset_size': 'medium',
            'n_genes': 50,
            'n_samples': 150,
            'method': 'lsco',
            'use_nestboot': False,
            'method_name': 'lsco_simple',
            'execution_time': 6.2,
            'memory_usage': 120.0,
            'parameter_value': 0.05,
            'parameter_name': 'threshold',
            'num_edges': 445,
            'density': 0.178,
            'sparsity': 0.822,
            'f1_score': 0.750,
            'mcc': 0.689,
            'sensitivity': 0.789,
            'specificity': 0.820,
            'precision': 0.714,
            'true_positives': 45,
            'false_positives': 18,
            'true_negatives': 25,
            'false_negatives': 12
        },
        {
            'timestamp': '2025-07-18T10:25:00',
            'dataset_size': 'medium',
            'n_genes': 50,
            'n_samples': 150,
            'method': 'lasso',
            'use_nestboot': True,
            'method_name': 'lasso_nestboot',
            'execution_time': 180.5,
            'memory_usage': 320.0,
            'parameter_value': 0.05,
            'parameter_name': 'FDR',
            'num_edges': 365,
            'density': 0.146,
            'sparsity': 0.854,
            'f1_score': 0.695,
            'mcc': 0.628,
            'sensitivity': 0.645,
            'specificity': 0.890,
            'precision': 0.755,
            'true_positives': 35,
            'false_positives': 11,
            'true_negatives': 33,
            'false_negatives': 21
        }
    ]
    
    # Save Python results
    python_df = pd.DataFrame(python_data)
    python_df.to_csv(output_dir / "python_benchmark_results.csv", index=False)
    
    # Create Python summary
    python_summary = {
        'LASSO': {
            'avg_execution_time': np.mean([r['execution_time'] for r in python_data if r['method'] == 'LASSO']),
            'avg_memory_usage': np.mean([r['memory_usage'] for r in python_data if r['method'] == 'LASSO']),
            'avg_f1_score': np.mean([r['f1_score'] for r in python_data if r['method'] == 'LASSO']),
            'avg_precision': np.mean([r['precision'] for r in python_data if r['method'] == 'LASSO']),
            'avg_recall': np.mean([r['sensitivity'] for r in python_data if r['method'] == 'LASSO']),
            'avg_sparsity': np.mean([r['sparsity'] for r in python_data if r['method'] == 'LASSO']),
            'avg_density': np.mean([r['density'] for r in python_data if r['method'] == 'LASSO'])
        },
        'LSCO': {
            'avg_execution_time': np.mean([r['execution_time'] for r in python_data if r['method'] == 'LSCO']),
            'avg_memory_usage': np.mean([r['memory_usage'] for r in python_data if r['method'] == 'LSCO']),
            'avg_f1_score': np.mean([r['f1_score'] for r in python_data if r['method'] == 'LSCO']),
            'avg_precision': np.mean([r['precision'] for r in python_data if r['method'] == 'LSCO']),
            'avg_recall': np.mean([r['sensitivity'] for r in python_data if r['method'] == 'LSCO']),
            'avg_sparsity': np.mean([r['sparsity'] for r in python_data if r['method'] == 'LSCO']),
            'avg_density': np.mean([r['density'] for r in python_data if r['method'] == 'LSCO'])
        }
    }
    
    with open(output_dir / "python_summary.json", 'w') as f:
        json.dump(python_summary, f, indent=2)
    
    # Save MATLAB results
    matlab_df = pd.DataFrame(matlab_data)
    matlab_df.to_csv(output_dir / "matlab_benchmark_results.csv", index=False)
    
    # Create MATLAB summary
    matlab_summary = {}
    for method_name in ['lasso_simple', 'lsco_simple', 'lasso_nestboot']:
        method_data = [r for r in matlab_data if r['method_name'] == method_name]
        if method_data:
            matlab_summary[method_name] = {
                'avg_execution_time': np.mean([r['execution_time'] for r in method_data]),
                'avg_memory_usage': np.mean([r['memory_usage'] for r in method_data]),
                'avg_f1_score': np.mean([r['f1_score'] for r in method_data]),
                'avg_precision': np.mean([r['precision'] for r in method_data]),
                'avg_recall': np.mean([r['sensitivity'] for r in method_data]),
                'avg_sparsity': np.mean([r['sparsity'] for r in method_data]),
                'avg_density': np.mean([r['density'] for r in method_data]),
                'count': len(method_data)
            }
    
    with open(output_dir / "matlab_summary.json", 'w') as f:
        json.dump(matlab_summary, f, indent=2)
    
    print(f"✅ Sample data created in {output_dir}")
    print(f"   📊 Python results: {len(python_data)} entries")
    print(f"   📊 MATLAB results: {len(matlab_data)} entries")
    
    return output_dir

def test_report_generation():
    """Test the report generation with sample data."""
    print("\n🧪 Testing report generation...")
    
    try:
        # Import the report generator
        from generate_performance_report import PerformanceReportGenerator
        
        # Create generator and run
        generator = PerformanceReportGenerator()
        generator.load_results()
        generator.analyze_performance()
        
        # Generate report without visualizations (to avoid matplotlib issues)
        report_content = generator.generate_markdown_report()
        
        # Save report
        output_file = generator.save_report("TEST_PERFORMANCE_REPORT.md")
        
        print(f"✅ Test report generated successfully!")
        print(f"📄 Report file: {output_file}")
        
        # Show a snippet of the report
        lines = report_content.split('\n')
        print(f"\n📝 Report preview (first 10 lines):")
        for i, line in enumerate(lines[:10]):
            print(f"   {i+1:2d}: {line}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the report generator script is available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function."""
    print("🚀 Performance Report Test Suite")
    print("=" * 40)
    
    # Create sample data
    output_dir = create_sample_data()
    
    # Test report generation
    success = test_report_generation()
    
    if success:
        print(f"\n🎉 All tests passed!")
        print(f"📁 Sample data available in: {output_dir}")
        print(f"🔧 You can now run: python generate_performance_report.py")
    else:
        print(f"\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
