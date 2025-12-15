import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.methods.nestboot import Nestboot, NetworkResults, AnalysisConfig
from src.datastruct.Dataset import Dataset
from src.analyze.Data import Data

class TestNestboot:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        Y = np.random.rand(5, 10)
        P = np.random.rand(5, 10)
        
        # Use a simple class instead of MagicMock to avoid spec issues with private attributes
        class MockDataset:
            def __init__(self):
                self.Y = Y
                self.P = P
                self.N = 5
                self.M = 10
                self._network = None
                self._names = [f"Gene_{i:02d}" for i in range(5)]
                self._E = None
                self._lambda = None
                self._dataset_name = "Test"
                self.data = None # To simulate it not being a Data wrapper itself
        
        dataset = MockDataset()
        
        # Mock Data wrapper
        data = MagicMock(spec=Data)
        data.data = dataset
        # Do not set Y, P, N, M on data mock directly, as Data object delegates or doesn't have them directly
        # This ensures nestboot logic picks data.data as ds_obj
        
        return data

    def test_nestboot_init(self):
        nb = Nestboot()
        assert isinstance(nb.config, AnalysisConfig)
        
        config = AnalysisConfig(total_runs=10)
        nb = Nestboot(config)
        assert nb.config.total_runs == 10
        
        nb = Nestboot({'total_runs': 20})
        assert nb.config.total_runs == 20

    def test_compute_assign_frac(self):
        nb = Nestboot()
        
        # Create synthetic dataframe
        data = {
            'gene_i': ['G1', 'G1', 'G2'],
            'gene_j': ['G2', 'G2', 'G3'],
            'run': [0, 1, 0],
            'link_value': [0.5, 0.6, 0.7]
        }
        df = pd.DataFrame(data)
        
        results = nb.compute_assign_frac(df, total_runs=2, inner_group_size=1)
        
        assert 'Afrac' in results.columns
        # G1-G2 appears in 2 runs out of 2 -> Afrac = 1.0
        # G2-G3 appears in 1 run out of 2 -> Afrac = 0.5
        
        g1g2 = results[(results['gene_i'] == 'G1') & (results['gene_j'] == 'G2')]
        assert g1g2['Afrac'].values[0] == 1.0
        
        g2g3 = results[(results['gene_i'] == 'G2') & (results['gene_j'] == 'G3')]
        assert g2g3['Afrac'].values[0] == 0.5

    def test_run_nestboot_mock(self, synthetic_data):
        nb = Nestboot({'fdr_threshold': 0.1})
        
        # Mock inference method
        # Returns a random adjacency matrix
        def mock_inference(dataset, **kwargs):
            return np.random.rand(5, 5)
            
        # Run nestboot with small number of runs
        # We need to patch Data and Dataset imports inside run_nestboot if they are imported there
        # But looking at the code, they are imported inside run_nestboot
        
        # We can mock the imports using patch.dict or just rely on the fact that we are passing objects
        # The code does: from datastruct.Dataset import Dataset
        # So we need to make sure that import works or is mocked if we want to control the object creation
        
        # However, since we are running in the same environment, the real imports should work.
        # The issue is that run_nestboot creates NEW Dataset objects.
        # We need to ensure those new objects work with our mock inference.
        
        # Our mock inference just takes whatever is passed and returns a matrix.
        # So it should be fine as long as the Dataset creation doesn't fail.
        
        results = nb.run_nestboot(
            dataset=synthetic_data,
            inference_method=mock_inference,
            nest_runs=2,
            boot_runs=2,
            seed=42
        )
        
        assert isinstance(results, NetworkResults)
        # The result xnet size depends on the number of genes found in the bootstrap data
        # Since we use "Gene_00" etc, and we have 5 genes, it should be 5x5 if all genes are found
        # But if some genes have no links in any bootstrap, they might be missing from the merged dataframe?
        # Actually compute_assign_frac groups by gene_i, gene_j.
        # And _compute_network_metrics creates xnet from merged dataframe.
        # The xnet returned is a numpy array.
        # Wait, NetworkResults.xnet is a numpy array.
        # But how is it constructed?
        # xnet = (merged['Afrac_norm'] >= support_threshold).astype(float)
        # This is a Series/Array, not a matrix.
        # Ah, looking at the code:
        # xnet=xnet.values
        # So it returns a 1D array of edges?
        # Let's check the code again.
        
        # In nestboot.py:
        # xnet = (merged['Afrac_norm'] >= support_threshold).astype(float)
        # ...
        # return NetworkResults(xnet=xnet.values, ...)
        
        # Yes, it seems to return arrays corresponding to the rows in the merged dataframe (edges).
        # It does NOT return a square adjacency matrix.
        
        assert isinstance(results.xnet, np.ndarray)
