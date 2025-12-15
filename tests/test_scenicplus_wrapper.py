import sys
from unittest.mock import MagicMock, patch

# Mock scanpy before importing src.methods
sys.modules['scanpy'] = MagicMock()

import pytest
import numpy as np
import pandas as pd
import os
from src.methods import run, SCENICPLUS
from src.datastruct.Dataset import Dataset

@pytest.fixture
def mock_dataset():
    """Create a dummy dataset for testing."""
    dataset = Dataset()
    # 10 genes, 5 samples
    dataset._Y = np.random.rand(10, 5)
    dataset._P = np.random.rand(10, 5) # Add P matrix with matching dimensions
    dataset._names = [f"Gene_{i}" for i in range(10)]
    return dataset

@patch('src.methods.scenicplus.subprocess.run')
@patch('src.methods.scenicplus.sc.AnnData.write_h5ad')
@patch('src.methods.scenicplus.os.path.exists') # Mock os.path.exists
def test_scenicplus_wrapper_success(mock_exists, mock_write_h5ad, mock_subprocess, mock_dataset, tmp_path):
    """Test SCENICPLUS wrapper success path with mocking."""
    
    # Mock subprocess.run to return success
    mock_subprocess.return_value = MagicMock(returncode=0, stdout="Success")
    
    # Mock os.path.exists to return True for config, snakefile, and results
    # We need to be careful not to break other checks.
    # The code checks:
    # 1. config_path (real file)
    # 2. snakefile_path (real file)
    # 3. results_file (generated file)
    
    def side_effect(path):
        if "eRegulons" in str(path):
            return True
        if "config.yaml" in str(path):
            return True
        if "Snakefile" in str(path):
            return True
        return False # Default to False for others if any
        
    mock_exists.side_effect = side_effect
    
    # We need to simulate the creation of the output file by Snakemake.
    # Since SCENICPLUS runs in a temp dir we don't control easily from outside without mocking tempfile,
    # we can mock pd.read_csv to return a dummy dataframe instead of relying on the file existing.
    
    with patch('src.methods.scenicplus.pd.read_csv') as mock_read_csv:
        # Mock the results dataframe
        mock_df = pd.DataFrame({
            'TF': ['Gene_0', 'Gene_1'],
            'Gene': ['Gene_1', 'Gene_2'],
            'importance': [0.5, 0.8]
        })
        mock_read_csv.return_value = mock_df
        
        # Run the function
        # We pass a dummy cisTopic path
        adj = run('scenicplus', mock_dataset, cisTopic_obj_fname="dummy.pkl")
        
        # Verify subprocess was called
        assert mock_subprocess.called
        args, _ = mock_subprocess.call_args
        cmd = args[0]
        assert "snakemake" in cmd
        assert "all" in cmd
        
        # Verify result shape
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 0.5
        assert adj[1, 2] == 0.8

@patch('src.methods.scenicplus.subprocess.run')
def test_scenicplus_wrapper_failure(mock_subprocess, mock_dataset):
    """Test SCENICPLUS wrapper failure handling."""
    
    # Mock subprocess.run to return failure
    mock_subprocess.return_value = MagicMock(returncode=1, stdout="Error", stderr="Failed")
    
    with pytest.raises(RuntimeError, match="Snakemake failed"):
        run('scenicplus', mock_dataset, cisTopic_obj_fname="dummy.pkl")

def test_unified_run_lasso(mock_dataset):
    """Test unified run with Lasso (mocked or real if fast)."""
    # Lasso is fast enough to run real, but let's mock to avoid dependency issues if any
    # We must patch 'src.methods.Lasso' because 'run' uses the imported symbol in src.methods
    with patch('src.methods.Lasso') as mock_lasso:
        mock_lasso.return_value = (np.zeros((10, 10)), np.array([0.1]))
        
        run('lasso', mock_dataset, alpha_range=np.array([0.5]))
        
        mock_lasso.assert_called_once()
        # Check args
        args, kwargs = mock_lasso.call_args
        assert args[0] == mock_dataset
        assert np.array_equal(kwargs['alpha_range'], np.array([0.5]))

@patch('src.methods.nestboot.Nestboot.run_nestboot')
def test_unified_run_nested_boot(mock_run_nestboot, mock_dataset):
    """Test unified run with nested_boot=True."""
    
    run('lasso', mock_dataset, nested_boot=True, nest_runs=10, boot_runs=5, fdr=0.1, alpha_range=np.array([0.5]))
    
    mock_run_nestboot.assert_called_once()
    _, kwargs = mock_run_nestboot.call_args
    
    assert kwargs['dataset'] == mock_dataset
    assert kwargs['nest_runs'] == 10
    assert kwargs['boot_runs'] == 5
    # Check that method_kwargs contains the alpha parameter
    assert np.array_equal(kwargs['method_kwargs']['alpha_range'], np.array([0.5]))
    
    # Check that the inference method passed is Lasso
    from src.methods.lasso import Lasso
    assert kwargs['inference_method'] == Lasso

