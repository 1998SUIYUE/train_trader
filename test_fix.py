#!/usr/bin/env python3
"""
Test script to verify the fixes for GPUDataProcessor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from pathlib import Path

# Test data creation
def create_test_data():
    """Create a small test dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate OHLCV data
    base_price = 1800
    prices = []
    current_price = base_price
    
    for i in range(n_samples):
        # Random walk
        change = np.random.normal(0, 0.01) * current_price
        current_price += change
        
        # Generate OHLC
        volatility = abs(np.random.normal(0, 0.005))
        high = current_price * (1 + volatility)
        low = current_price * (1 - volatility)
        open_price = current_price + np.random.normal(0, 0.003) * current_price
        close_price = current_price
        
        # Ensure OHLC logic
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Volume
        volume = np.random.randint(10000, 50000)
        
        prices.append([open_price, high, low, close_price, volume])
    
    # Create DataFrame
    df = pd.DataFrame(prices, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    return df

def test_gpu_data_processor():
    """Test GPUDataProcessor with our fixes"""
    print("=== Testing GPUDataProcessor Fixes ===")
    
    try:
        # Import modules
        from gpu_utils import get_windows_gpu_manager
        from data_processor import GPUDataProcessor
        
        print("✅ Imports successful")
        
        # Create test data
        df = create_test_data()
        test_file = 'test_data_temp.csv'
        df.to_csv(test_file, index=False)
        print(f"✅ Test data created: {df.shape}")
        
        # Initialize GPU manager
        gpu_manager = get_windows_gpu_manager()
        print(f"✅ GPU manager initialized: {gpu_manager.device}")
        
        # Test GPUDataProcessor initialization with correct parameter name
        processor = GPUDataProcessor(
            window_size=50,  # Small window for testing
            normalization_method='relative',
            gpu_manager=gpu_manager
        )
        print("✅ GPUDataProcessor initialized successfully")
        
        # Test load_and_process_data method
        features, labels = processor.load_and_process_data(test_file)
        print(f"✅ Data processing successful")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        print("✅ Test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if Path(test_file).exists():
            Path(test_file).unlink(missing_ok=True)

if __name__ == "__main__":
    success = test_gpu_data_processor()
    if success:
        print("\n🎉 All fixes are working correctly!")
        sys.exit(0)
    else:
        print("\n💥 Some issues remain")
        sys.exit(1)