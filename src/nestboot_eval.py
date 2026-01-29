
import sys, os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
sys.path.insert(0, 'src')

from analyze.Data import Data
from datastruct.Network import Network
from analyze.CompareModels import CompareModels
from methods.lasso import Lasso
from methods.lsco import LSCO
from methods.genie3 import GENIE3
from methods.clr import CLR
from methods.tigress import TIGRESS
from methods.nestboot import Nestboot

zetavec = np.logspace(-6, 0, 30)
SNR=['10','1000','100000']
methods = [Lasso, LSCO, CLR, GENIE3, TIGRESS]
IDs=['436458', '252384', '5072679','5463023','412416','453556','461522','4941145','5463023','4591182','5072679','4982479']

# 2. Run Inference Methods
nb = Nestboot()

def process_single_run(snr, id, method, zetavec, nb):
    try:
        dataset = Data.from_json_url(
            'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID'+id+'-D20151111-N50-E150-SNR'+snr+'-IDY'+id+'.json'
        )

        true_net = Network.from_json_url(
            'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/'+dataset.data.network.network.split('-')[2]+'/N50/'+dataset.data.network.network+'.json'
        )
        
        topology = dataset.data.network.network.split('-')[2]
        results_list = []

        print(f"Running {method.__name__} with SNR={snr}, ID={id}")
        
        # 1. Base Method
        if method == Lasso:
            net, _ = method(dataset, alpha_range=zetavec)
        elif method == LSCO:
            net, _ = method(dataset, threshold_range=zetavec)
        elif method == TIGRESS:
             # TIGRESS needs R and L
             net, _ = method(dataset)
        else:
            net, _ = method(dataset)
            
        results = CompareModels(true_net, net)
        results_list.append({
            'Method': method.__name__,
            'SNR': snr,
            'nestboot': 'No',
            'topology': topology,
            'random': 'No',
            'Max_AUROC': max(results.AUROC),
            'Max_MCC': max(results.MCC),
            'Max_F1': max(results.F1)
        })

        # 2. Randomized Data
        n_samples = dataset.data.Y.shape[1]
        n_genes = dataset.data.Y.shape[0]
        randomized_data = Data.from_json_url(
            'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID'+id+'-D20151111-N50-E150-SNR'+snr+'-IDY'+id+'.json'
        )

        shuffle_indices_ya = np.random.permutation(n_samples)
        shuffle_indices_yb = np.random.permutation(n_genes)
                
        # IMPORTANT: Fix bug in original script (it was running shuffled_data.Y = ... then shuffled_data.Y = ... sequentially without applying both)
        Y_shuf = dataset.data.Y.copy()
        Y_shuf = Y_shuf[:, shuffle_indices_ya]
        Y_shuf = Y_shuf[shuffle_indices_yb, :]
        randomized_data.data.Y = Y_shuf

        if method == Lasso:
            net, _ = method(randomized_data, alpha_range=zetavec)
        elif method == LSCO:
            net, _ = method(randomized_data, threshold_range=zetavec)
        elif method == TIGRESS:
            net, _ = method(randomized_data)
        else:
            net, _ = method(randomized_data)

        results = CompareModels(true_net, net)
        results_list.append({
            'Method': method.__name__,
            'SNR': snr,
            'nestboot': 'No',
            'topology': topology,
            'random': 'Yes',
            'Max_AUROC': max(results.AUROC),
            'Max_MCC': max(results.MCC),
            'Max_F1': max(results.F1)
        })

        # 3. NestBoot
        # Prepare method_params
        m_params = {}
        if method == Lasso:
             m_params = {'alpha_range': zetavec}
        elif method == LSCO:
             m_params = {'threshold_range': zetavec}
            
        results0 = nb.run_nestboot(
            dataset=dataset,
            inference_method=method, 
            method_params=m_params,
            nest_runs=5,
            boot_runs=5,
            seed=42
        )
        
        # Use results0.sxnet (continuous) for AUROC
        # results0.xnet (binary) for F1/MCC? 
        # CompareModels expects continuous for AUROC.
        # Original code used xnet for all. Let's fix that.
        
        # Fix: Use sxnet for AUROC if available, else xnet
        metric_net = results0.sxnet if results0.sxnet is not None else results0.xnet
        
        # Nestboot result might be a list if multiple params.
        # But run_nestboot _compute_network_metrics might return list for sxnet?
        # Let's assume Nestboot outputs are lists if multiple parameters, but single array if aggregated?
        # Check Nestboot implementation... it seems to return single NetworkResults object
        # where xnet, sxnet are lists if multiple params? No, standard usage seems to be single aggregate? 
        # Actually in nestboot.py: xnet_list.append(xnet)... return NetworkResults(xnet=xnet, ...) 
        
        # Safe handling
        if isinstance(metric_net, list):
             # Choose best or first? Usually standard Nestboot combines them or we pick best.
             # Let's take max over all
             aurocs = []
             mccs = []
             f1s = []
             for i in range(len(metric_net)):
                 cm = CompareModels(true_net, metric_net[i])
                 aurocs.append(max(cm.AUROC))
                 mccs.append(max(cm.MCC))
                 f1s.append(max(cm.F1))
             max_auroc = max(aurocs)
             max_mcc = max(mccs)
             max_f1 = max(f1s)
        else:
             cm = CompareModels(true_net, metric_net)
             max_auroc = max(cm.AUROC)
             max_mcc = max(cm.MCC)
             max_f1 = max(cm.F1)
             
        results_list.append({
            'Method': method.__name__,
            'SNR': snr,
            'nestboot': 'yes',
            'topology': topology,
            'random': 'No',
            'Max_AUROC': max_auroc,
            'Max_MCC': max_mcc,
            'Max_F1': max_f1
        })
        
        return results_list
        
    except Exception as e:
        print(f"Error processing {method.__name__} SNR={snr} ID={id}: {e}")
        return []

# Flatten task list
tasks = []
for snr in SNR:
    for id in IDs:
        for method in methods:
            tasks.append((snr, id, method))

# Run in parallel
print(f"Starting parallel execution of {len(tasks)} tasks...")
all_results = Parallel(n_jobs=-1)(
    delayed(process_single_run)(snr, id, method, zetavec, nb) 
    for snr, id, method in tasks
)

# Aggregate and save
flat_results = [item for sublist in all_results for item in sublist]
df = pd.DataFrame(flat_results)
df.to_csv('nestboot_evaluation_results.csv', index=False)
print("Done. Saved to nestboot_evaluation_results.csv")


