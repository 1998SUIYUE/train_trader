#!/usr/bin/env python3
"""
Simple CUDA Environment Test
Tests basic CUDA functionality without complex imports
"""

import sys
import time

def test_basic_imports():
    """Test basic library imports"""
    print("=== Basic Library Import Test ===")
    
    try:
        import torch
        print(f"SUCCESS: PyTorch {torch.__version__}")
        return True
    except ImportError as e:
        print(f"ERROR: PyTorch import failed: {e}")
        return False

def test_cuda_basic():
    """Test basic CUDA functionality"""
    print("\n=== Basic CUDA Test ===")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available")
            print("Possible reasons:")
            print("  1. CUDA not installed")
            print("  2. PyTorch version doesn't support current CUDA")
            print("  3. NVIDIA drivers too old")
            return False
        
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        # Show GPU details
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: CUDA test failed: {e}")
        return False

def test_cuda_computation():
    """Test CUDA computation"""
    print("\n=== CUDA Computation Test ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("SKIPPED: CUDA not available")
            return False
        
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        
        # Test matrix multiplication
        size = 1000
        print(f"Testing {size}x{size} matrix multiplication")
        
        # CPU test
        x_cpu = torch.randn(size, size)
        y_cpu = torch.randn(size, size)
        
        start_time = time.time()
        z_cpu = torch.matmul(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.4f}s")
        
        # GPU test
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        
        # Warmup
        _ = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        
        start_time = time.time()
        z_gpu = torch.matmul(x_gpu, y_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        print(f"GPU time: {gpu_time:.4f}s")
        
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify results
        z_gpu_cpu = z_gpu.cpu()
        if torch.allclose(z_cpu, z_gpu_cpu, rtol=1e-4):
            print("SUCCESS: Results match")
            return True
        else:
            print("ERROR: Results don't match")
            return False
            
    except Exception as e:
        print(f"ERROR: Computation test failed: {e}")
        return False

def test_memory_management():
    """Test GPU memory management"""
    print("\n=== GPU Memory Management Test ===")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("SKIPPED: CUDA not available")
            return False
        
        device = torch.device('cuda:0')
        
        # Show initial memory
        allocated = torch.cuda.memory_allocated(device) / 1e9
        cached = torch.cuda.memory_reserved(device) / 1e9
        print(f"Initial memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        # Allocate memory
        print("Allocating GPU memory...")
        tensors = []
        for i in range(5):
            tensor = torch.randn(1000, 1000, device=device)
            tensors.append(tensor)
            
            allocated = torch.cuda.memory_allocated(device) / 1e9
            print(f"  After tensor {i+1}: {allocated:.2f}GB")
        
        # Clean up
        print("Cleaning up GPU memory...")
        del tensors
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated(device) / 1e9
        cached = torch.cuda.memory_reserved(device) / 1e9
        print(f"After cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
        
        print("SUCCESS: Memory management test passed")
        return True
        
    except Exception as e:
        print(f"ERROR: Memory test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Simple CUDA Environment Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("CUDA Basic", test_cuda_basic),
        ("CUDA Computation", test_cuda_computation),
        ("Memory Management", test_memory_management),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"ERROR: {test_name} test exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nSUCCESS: All tests passed! Your CUDA environment is ready.")
        print("\nNext steps:")
        print("  1. Run: python core/main_cuda.py")
        print("  2. Or try: python demo_cuda_training.py")
    elif passed >= total - 1:
        print("\nWARNING: Most tests passed, environment is mostly functional.")
        print("Some advanced features may not work, but basic training should be fine.")
    else:
        print("\nERROR: Multiple tests failed, please check environment setup.")
        print("\nSuggestions:")
        print("  1. Run installation script: powershell setup/install_cuda129.ps1")
        print("  2. Check CUDA and PyTorch version compatibility")
        print("  3. Update NVIDIA drivers")

if __name__ == "__main__":
    main()