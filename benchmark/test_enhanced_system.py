#!/usr/bin/env python3
"""
Simple test to verify enhanced performance system data loading works with uv.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_data_loading():
    """Test data loading functionality."""
    print("🧪 Testing data loading...")
    
    try:
        # Test importing the modules
        print("   📦 Testing imports...")
        from analyze.Data import Data
        from datastruct.Network import Network
        print("   ✅ Imports successful")
        
        # Test loading data from URL
        print("   📥 Testing data loading...")
        dataset_url = "https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json"
        
        data_obj = Data.from_json_url(dataset_url)
        dataset = data_obj.data
        
        print(f"   ✅ Dataset loaded: {dataset.dataset}")
        print(f"      📊 Expression matrix: {dataset.Y.shape}")
        print(f"      🧬 Genes: {dataset.Y.shape[0]}, 🔬 Samples: {dataset.Y.shape[1]}")
        
        # Test loading network
        print("   📥 Testing network loading...")
        network_url = "https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json"
        
        network = Network.from_json_url(network_url)
        print(f"   ✅ Network loaded: {network.network}")
        print(f"      🔗 Network shape: {network.A.shape}")
        print(f"      📈 Edges: {(network.A != 0).sum()}")
        
        # Test LASSO function
        print("   🔍 Testing LASSO...")
        from methods.lasso import Lasso
        inferred_A, alpha = Lasso(dataset)
        print(f"   ✅ LASSO completed: alpha={alpha:.6f}, network shape={inferred_A.shape}")
        
        # Test LSCO function
        print("   🔍 Testing LSCO...")
        from methods.lsco import LSCO
        inferred_A2, mse = LSCO(dataset)
        print(f"   ✅ LSCO completed: MSE={mse:.6f}, network shape={inferred_A2.shape}")
        
        print("\n🎉 All tests passed! Enhanced system ready to run.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
