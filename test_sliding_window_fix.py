#!/usr/bin/env python3
"""
Test script to verify the sliding window fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path

def test_sliding_window_fix():
    """Test the sliding window fix"""
    print("=== Testing Sliding Window Fix ===")
    
    try:
        # Import modules
        from gpu_utils import get_windows_gpu_manager
        from data_processor import GPUDataProcessor
        
        print("✅ Imports successful")
        
        # Check if sample data exists
        data_file = "XAUUSD_M1_202503142037_202506261819.csv"
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False
        
        print(f"✅ Data file found: {data_file}")
        
        # Initialize GPU manager
        gpu_manager = get_windows_gpu_manager()
        print(f"✅ GPU manager initialized: {gpu_manager.device}")
        
        # Test GPUDataProcessor with small window size
        processor = GPUDataProcessor(
            window_size=20,  # Small window for testing
            normalization_method='relative',
            gpu_manager=gpu_manager
        )
        print("✅ GPUDataProcessor initialized successfully")
        
        # Test load_and_process_data method
        features, labels = processor.load_and_process_data(data_file)
        print(f"✅ Data processing successful")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Features dtype: {features.dtype}")
        print(f"   Labels dtype: {labels.dtype}")
        
        # Test with different normalization methods
        for method in ['relative', 'rolling']:
            try:
                processor_test = GPUDataProcessor(
                    window_size=20,
                    normalization_method=method,
                    gpu_manager=gpu_manager
                )
                features_test, labels_test = processor_test.load_and_process_data(data_file)
                print(f"✅ {method} normalization successful: {features_test.shape}")
            except Exception as e:
                print(f"❌ {method} normalization failed: {e}")
        
        print("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sliding_window_fix()
    if success:
        print("\n🎉 Sliding window fix is working correctly!")
        sys.exit(0)
    else:
        print("\n💥 Issues still remain")
        sys.exit(1)