#!/usr/bin/env python3
"""
Minimal test to verify the fixes work
"""

import sys
import os
import subprocess

def run_minimal_test():
    """Run a minimal test with very small parameters"""
    try:
        # Check if data file exists
        data_file = "XAUUSD_M1_202503142037_202506261819.csv"
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False
        
        print(f"✅ Data file found: {data_file}")
        
        # Run the GPU training with minimal parameters
        cmd = [
            "python", "core/main_gpu.py",
            "--data_file", data_file,
            "--population_size", "5",
            "--generations", "2",
            "--window_size", "20"
        ]
        
        print(f"🚀 Running command: {' '.join(cmd)}")
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("📊 Output:")
            print(result.stdout[-500:])  # Last 500 characters
            return True
        else:
            print("❌ Training failed!")
            print("📊 Error output:")
            print(result.stderr[-500:])  # Last 500 characters
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    print("=== Minimal Test for GPU Training Fixes ===")
    success = run_minimal_test()
    
    if success:
        print("\n🎉 All fixes are working correctly!")
        sys.exit(0)
    else:
        print("\n💥 Issues still remain")
        sys.exit(1)