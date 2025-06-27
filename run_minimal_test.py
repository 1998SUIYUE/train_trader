#!/usr/bin/env python3
"""
Run minimal training test with very small parameters
"""

import subprocess
import sys
import os

def run_minimal_training():
    """Run minimal training test"""
    print("=== Running Minimal Training Test ===")
    
    # Check if data file exists
    data_file = "XAUUSD_M1_202503142037_202506261819.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    print(f"✅ Data file found: {data_file}")
    
    # Run minimal training
    cmd = [
        sys.executable, "core/main_gpu.py",
        "--data_file", data_file,
        "--population_size", "5",
        "--generations", "2", 
        "--window_size", "20"
    ]
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print("📊 Last 500 characters of output:")
            print(result.stdout[-500:])
            return True
        else:
            print("❌ Training failed!")
            print("📊 Error output:")
            print(result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Training timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"❌ Training failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = run_minimal_training()
    if success:
        print("\n🎉 Minimal training test successful!")
    else:
        print("\n💥 Training test failed")
    
    input("\nPress Enter to continue...")