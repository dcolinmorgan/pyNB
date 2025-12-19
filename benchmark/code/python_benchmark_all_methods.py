import sys
import os
import json
import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, 'src')
from analyze.Data import Data
from datastruct.Network import Network
from methods.lasso import Lasso
from methods.lsco import LSCO
from methods.clr import CLR
from methods.genie3 import GENIE3
from methods.tigress import TIGRESS
from analyze.CompareModels import CompareModels

def run_comparison_analysis(true_net, inferred_net):
    """Compare inferred network with true network."""
    comp = CompareModels(true_net, Network(inferred_net))

    try:
        from sklearn.metrics import roc_auc_score
        true_flat = (true_net.A != 0).astype(float).flatten()
        inferred_flat = np.abs(inferred_net.flatten())

        if len(np.unique(true_flat)) > 1 and np.sum(inferred_flat) > 0:
            auroc = roc_auc_score(true_flat, inferred_flat)
        else:
            auroc = 0.5
    except:
        auroc = 0.5

    return {
        'f1_score': comp.F1[0] if len(comp.F1) > 0 else 0.0,
        'auroc': auroc,
        'precision': comp.pre[0] if len(comp.pre) > 0 else 0.0,
        'recall': comp.sen[0] if len(comp.sen) > 0 else 0.0,
        'mcc': comp.MCC[0] if len(comp.MCC) > 0 else 0.0
    }

def find_network_file(network_dir, network_id):
    """Find network file recursively with the given ID."""
    # Look for files containing the ID
    matches = list(Path(network_dir).rglob(f"*{network_id}*.json"))

    if matches:
        return matches[0]

    # Try different patterns if not found
    patterns = [
        f"**/*ID{network_id}.json",
        f"**/*ID{network_id}*",
        f"**/*{network_id}.json",
        f"**/*{network_id}*"
    ]

    for pattern in patterns:
        matches = list(Path(network_dir).rglob(pattern))
        if matches:
            return matches[0]

    return None

def run_method(method_name, data, zetavec):
    """Run a specific inference method."""
    start_time = time.time()

    try:
        if method_name == 'LASSO':
            A_3d, _ = Lasso(data, alpha_range=zetavec)
            # Take the last (most sparse) result
            A_final = A_3d[:, :, -1]
        elif method_name == 'LSCO':
            A_3d, _ = LSCO(data, threshold_range=zetavec)
            A_final = A_3d[:, :, -1]
        elif method_name == 'CLR':
            A_final = CLR(data)
        elif method_name == 'GENIE3':
            A_final = GENIE3(data)
        elif method_name == 'TIGRESS':
            A_final = TIGRESS(data)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        exec_time = time.time() - start_time
        return A_final, exec_time, None  # No memory tracking in Python

    except Exception as e:
        print(f"   ❌ Error running {method_name}: {e}")
        exec_time = time.time() - start_time
        return None, exec_time, str(e)

def main():
    # Configuration
    OUTPUT_DIR = 'benchmark_results'
    FIGURES_DIR = 'benchmark_figures'

    # Create output directories
    output_dir = Path(OUTPUT_DIR)
    figures_dir = Path(FIGURES_DIR)
    output_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Dataset and network paths
    dataset_root = os.path.expanduser('../GeneSPIDER2/data/gs-datasets/N50')
    network_root = os.path.expanduser('../GeneSPIDER2/data/gs-networks')

    # Zetavec for LASSO and LSCO
    zetavec = np.logspace(-6, 0, 30)

    print("🚀 Python All-Methods N50 Benchmark")
    print("=" * 50)

    # Find all N50 datasets
    dataset_files = sorted(glob.glob(os.path.join(dataset_root, "*.json")))
    print(f"Found {len(dataset_files)} N50 datasets.")

    # Results storage
    all_results = []
    results_file = output_dir / 'python_all_methods_results.csv'

    # Process all datasets
    processed_count = 0

    for dataset_path in dataset_files:
        dataset_filename = os.path.basename(dataset_path)
        print(f"\n🔄 [{processed_count+1}/{len(dataset_files)}] Processing {dataset_filename}")

        try:
            # Load dataset
            data = Data.from_json_file(dataset_path)

            # Extract network ID
            network_id = None
            if hasattr(data.data, 'network') and data.data.network:
                network_id = data.data.network.network.split('-ID')[-1]
            else:
                with open(dataset_path, 'r') as f:
                    json_data = json.load(f)
                if 'obj_data' in json_data and 'network' in json_data['obj_data']:
                    network_id = json_data['obj_data']['network'].split('-ID')[-1]

            if not network_id:
                print(f"   ⚠️ Could not extract network ID from {dataset_filename}")
                continue

            # Find network file
            network_path = find_network_file(network_root, network_id)
            if not network_path:
                print(f"   ⚠️ Network file not found for ID {network_id}")
                continue

            # Load network
            net = Network.from_json_file(str(network_path))

            print(f"   📂 Network: {os.path.basename(str(network_path))}")

            # Run all methods
            methods = ['LASSO', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS']

            for method in methods:
                print(f"   Running {method}...")
                inferred_net, exec_time, error = run_method(method, data, zetavec)

                if inferred_net is not None:
                    # Compare with true network
                    metrics = run_comparison_analysis(net, inferred_net)

                    # Store results
                    result = {
                        'dataset': dataset_filename,
                        'network': os.path.basename(str(network_path)),
                        'method': method,
                        'execution_time': exec_time,
                        'memory_usage': 0.0,  # Not tracked
                        'f1_score': metrics['f1_score'],
                        'auroc': metrics['auroc'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'mcc': metrics['mcc'],
                        'density': np.sum(inferred_net != 0) / inferred_net.size,
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(result)

                    print(".3f")
                else:
                    print(f"   ❌ {method} failed: {error}")

            # Save results after each dataset
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(results_file, index=False)

            processed_count += 1

        except Exception as e:
            print(f"   ❌ Error processing {dataset_filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n🏁 Benchmark Complete. Processed {processed_count} datasets.")
    print(f"   Results saved to: {results_file}")

    # Generate summary
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n📊 Summary by Method:")
        for method in methods:
            method_results = df_results[df_results['method'] == method]
            if len(method_results) > 0:
                print(f"   {method}:")
                print(f"      F1: {method_results['f1_score'].mean():.3f} ± {method_results['f1_score'].std():.3f}")
                print(f"      AUROC: {method_results['auroc'].mean():.3f} ± {method_results['auroc'].std():.3f}")
                print(f"      Time: {method_results['execution_time'].mean():.1f} ± {method_results['execution_time'].std():.1f}s")

if __name__ == "__main__":
    main()
