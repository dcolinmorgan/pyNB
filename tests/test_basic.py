import numpy as np
from datastruct.Network import Network
from datastruct.Experiment import Experiment
from datastruct.Dataset import Dataset
from analyse.Model import Model
from analyse.Data import Data
from analyse.CompareModels import CompareModels

def test_basic_functionality():
    # Create a simple network
    A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    net = Network(A)
    print(f"Network created: {net.network}")

    # Create an experiment
    exp = Experiment(net)
    exp.gaussian()  # Add some noise
    print(f"Experiment Y: {exp.noiseY()}")

    # Create a dataset
    ds = Dataset(exp)
    print(f"Dataset Y: {ds.Y}")

    # Analyze the network model
    model = Model(net)
    print(f"Condition Number: {model.interampatteness}")
    print(f"Small-worldness: {model.proximity_ratio}")

    # Analyze the dataset
    data = Data(ds)
    print(f"SNR_L: {data.SNR_L}")

    # Compare models
    Alist = np.stack([A * 0.9, A * 1.1], axis=2)  # Two slightly perturbed versions
    comp = CompareModels(net, Alist)
    print(f"Absolute Frobenius Norm: {comp.afronorm}")
    print(f"F1 Score: {comp.F1}")

if __name__ == "__main__":
    test_basic_functionality()
